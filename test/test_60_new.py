import os

os.environ["CZSC_USE_PYTHON"] = "True"

from pathlib import Path

import pandas as pd
from loguru import logger

import czsc
from czsc import Event, Position
from czsc.connectors import research


def create_position(symbol, name, opens, exits=None, interval=3600 * 4, timeout=16 * 20, stop_loss=500):
    """Create a simple long-short position definition."""
    exits = exits or []
    return Position(
        name=name,
        symbol=symbol,
        opens=[Event.load(x) for x in opens],
        exits=[Event.load(x) for x in exits],
        interval=interval,
        timeout=timeout,
        stop_loss=stop_loss,
    )


def create_bi_trend_position(symbol, base_freq="60分钟"):
    opens = [
        {
            "name": "笔方向开多",
            "operate": "开多",
            "signals_all": [f"{base_freq}_D1_表里关系V230101_向上_任意_任意_0"],
            "signals_any": [],
            "signals_not": [f"{base_freq}_D1_涨跌停V230331_涨停_任意_任意_0"],
        },
        {
            "name": "笔方向开空",
            "operate": "开空",
            "signals_all": [f"{base_freq}_D1_表里关系V230101_向下_任意_任意_0"],
            "signals_any": [],
            "signals_not": [f"{base_freq}_D1_涨跌停V230331_跌停_任意_任意_0"],
        },
    ]
    return create_position(symbol, name=f"{base_freq}笔方向跟随", opens=opens, timeout=16 * 20)


def create_macd_link_position(symbol):
    opens = [
        {
            "name": "日60共振开多",
            "operate": "开多",
            "signals_all": ["日线#60分钟_MACD交叉_联立V230518_看多_任意_任意_0"],
            "signals_any": [],
            "signals_not": [],
        },
        {
            "name": "日60共振开空",
            "operate": "开空",
            "signals_all": ["日线#60分钟_MACD交叉_联立V230518_看空_任意_任意_0"],
            "signals_any": [],
            "signals_not": [],
        },
    ]
    return create_position(symbol, name="日线60分钟MACD共振", opens=opens, timeout=16 * 15)


def create_double_ma_position(symbol, base_freq="60分钟"):
    opens = [
        {
            "name": "双均线开多",
            "operate": "开多",
            "signals_all": [f"{base_freq}_D1N5M21双均线_BS辅助V240208_多头_任意_任意_0"],
            "signals_any": [],
            "signals_not": [],
        },
        {
            "name": "双均线开空",
            "operate": "开空",
            "signals_all": [f"{base_freq}_D1N5M21双均线_BS辅助V240208_空头_任意_任意_0"],
            "signals_any": [],
            "signals_not": [],
        },
    ]
    return create_position(symbol, name=f"{base_freq}双均线结构", opens=opens, timeout=16 * 20)


class Strategy(czsc.CzscStrategyBase):
    @property
    def positions(self):
        return [
            create_bi_trend_position(self.symbol, base_freq="60分钟"),
            create_macd_link_position(self.symbol),
            create_double_ma_position(self.symbol, base_freq="60分钟"),
        ]


def summarize_records(df):
    agg = (
        df.groupby("strategy")
        .agg(
            sample_size=("symbol", "nunique"),
            avg_cum_bp=("累计收益", "mean"),
            median_cum_bp=("累计收益", "median"),
            avg_trade_count=("交易次数", "mean"),
            avg_win_rate=("交易胜率", "mean"),
            avg_sharpe=("夏普", "mean"),
            avg_calmar=("卡玛", "mean"),
            avg_max_drawdown=("最大回撤", "mean"),
            positive_ratio=("累计收益", lambda x: round((x > 0).mean(), 4)),
        )
        .reset_index()
    )
    return agg.sort_values(["positive_ratio", "avg_cum_bp", "avg_sharpe"], ascending=False)


def build_observations(df_all, df_dir):
    lines = []
    top = df_all.iloc[0]
    lines.append(
        f"整体最稳的是 {top['strategy']}，正收益样本占比 {top['positive_ratio']:.2%}，平均累计收益 {top['avg_cum_bp']:.2f} BP。"
    )

    pivot = df_dir.pivot_table(index="strategy", columns="trade_dir", values="累计收益", aggfunc="mean").fillna(0)
    if {"多头", "空头"}.issubset(set(pivot.columns)):
        bias = (pivot["多头"] - pivot["空头"]).sort_values(ascending=False)
        long_best = bias.index[0]
        lines.append(f"{long_best} 明显更偏多头驱动，平均多头收益比空头高 {bias.iloc[0]:.2f} BP。")
        if bias.iloc[-1] > 0:
            lines.append(f"三套策略里都更吃多头行情，但 {bias.index[-1]} 的多空差距相对最小。")
        else:
            lines.append(f"{bias.index[-1]} 的空头端相对更占优，说明它更像一套对下跌更敏感的规则。")

    trades = df_all.sort_values("avg_trade_count", ascending=False).iloc[0]
    lines.append(
        f"交易最频繁的是 {trades['strategy']}，平均每个样本约 {trades['avg_trade_count']:.1f} 笔，"
        "更像高换手跟随策略。"
    )
    return lines


def run_research(results_path, symbol_count=12):
    symbols = research.get_symbols("中证500成分股")[:symbol_count]
    logger.info(f"研究样本数量：{len(symbols)}；样本列表前五个：{symbols[:5]}")

    records_all = []
    records_dir = []

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] 回测 {symbol}")
        tactic = Strategy(symbol=symbol, is_stocks=True)
        bars = research.get_raw_bars(symbol, freq=tactic.base_freq, sdt="20200101", edt="20220101")
        if not bars:
            logger.warning(f"{symbol} 无可用K线，跳过")
            continue

        trader = tactic.backtest(bars, sdt="20210101")
        for position in trader.positions:
            res_all = position.evaluate("多空")
            res_all.update({"symbol": symbol, "strategy": position.name})
            records_all.append(res_all)

            for trade_dir in ["多头", "空头"]:
                res_dir = position.evaluate(trade_dir)
                res_dir.update({"symbol": symbol, "strategy": position.name, "trade_dir": trade_dir})
                records_dir.append(res_dir)

    df_all = pd.DataFrame(records_all)
    df_dir = pd.DataFrame(records_dir)
    if df_all.empty:
        raise ValueError("没有生成任何回测结果，请检查数据路径和样本范围。")

    summary_all = summarize_records(df_all)
    summary_dir = (
        df_dir.groupby(["strategy", "trade_dir"])
        .agg(
            avg_cum_bp=("累计收益", "mean"),
            avg_trade_count=("交易次数", "mean"),
            avg_win_rate=("交易胜率", "mean"),
            avg_sharpe=("夏普", "mean"),
            avg_max_drawdown=("最大回撤", "mean"),
        )
        .reset_index()
        .sort_values(["strategy", "trade_dir"])
    )

    df_all.to_excel(results_path / "per_symbol_all.xlsx", index=False)
    df_dir.to_excel(results_path / "per_symbol_direction.xlsx", index=False)
    summary_all.to_excel(results_path / "summary_all.xlsx", index=False)
    summary_dir.to_excel(results_path / "summary_direction.xlsx", index=False)

    observations = build_observations(summary_all, df_dir)
    with open(results_path / "observations.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(observations))

    logger.info("整体汇总：\n{}", summary_all.to_string(index=False))
    logger.info("方向拆分汇总：\n{}", summary_dir.to_string(index=False))
    for line in observations:
        logger.info(f"观察结论：{line}")

    best_strategy_name = summary_all.iloc[0]["strategy"]
    best_symbol = (
        df_all[df_all["strategy"] == best_strategy_name].sort_values(["累计收益", "夏普"], ascending=False).iloc[0]["symbol"]
    )
    sample_symbol = best_symbol
    tactic = Strategy(symbol=sample_symbol, is_stocks=True)
    bars = research.get_raw_bars(sample_symbol, freq=tactic.base_freq, sdt="20200101", edt="20220101")
    replay_path = results_path / "replay_best_sample"
    trader = tactic.replay(bars, sdt="20210101", res_path=replay_path, refresh=True)
    logger.info(f"回放样本：{sample_symbol}；优先观察策略：{best_strategy_name}；持仓数量：{len(trader.positions)}")

    return summary_all, summary_dir


if __name__ == "__main__":
    results_path = Path("/Users/apple/Downloads/CZSC投研数据/策略研究/test_60_new")
    results_path.mkdir(exist_ok=True, parents=True)
    logger.add(results_path / "czsc.log", rotation="1 week", encoding="utf-8")

    summary_all, summary_dir = run_research(results_path=results_path, symbol_count=12)
    print(summary_all)
    print(summary_dir)
