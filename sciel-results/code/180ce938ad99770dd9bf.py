"""Market making with inventory risk: structural change via analytic hedging with time-varying depth."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill


def _check_deadline():
    """Return True if we should abort."""
    return time.time() > _DEADLINE


def quote(t, inventory, mid, params):
    """
    Quote bid/ask depths using an analytic hedging heuristic.
    
    The key insight: when inventory q is positive (long), we want to:
    - Widen the bid (reduce bid_depth to make it less attractive) to avoid more buys
    - Widen the ask (reduce ask_depth... wait, actually we want to make ask more attractive)
    
    Actually, let's think about this more carefully:
    - If q > 0 (long inventory), we want to sell, so we should make the ask more attractive (smaller ask_depth) and make the bid less attractive (larger bid_depth)
    - If q < 0 (short inventory), we want to buy, so we should make the bid more attractive (smaller bid_depth) and make the ask less attractive (larger ask_depth)
    
    The inventory penalty is (gamma/2) * sigma^2 * q^2 * dt per step.
    The marginal cost of inventory is gamma * sigma^2 * q * dt.
    
    We can use a linear hedging strategy:
    - bid_depth = base_depth + alpha * q
    - ask_depth = base_depth - alpha * q
    
    where base_depth is the risk-neutral depth and alpha captures the hedging pressure.
    
    We also need to ensure depths are positive and within valid bounds.
    The constraint is depth > -3/k (but since we want positive depths, this is usually satisfied).
    Actually, depth must be > -3/k, but for meaningful quotes we want depth > 0.
    
    Let's also consider time-to-horizon: as t approaches T, we should be more aggressive about
    reducing inventory, so alpha should increase as (T - t) decreases.
    
    For the base depth, the risk-neutral depth that balances expected fill rates is 1/k.
    But we can adjust this based on the regime parameters.
    
    Let's derive alpha more carefully:
    - Expected inventory change per step from bid fills: -lambda_bid (each bid fill reduces q by 1... wait)
    
    Actually, let's reconsider the inventory dynamics:
    - A bid fill: you sell at bid, so inventory decreases by 1 (you sold a unit)
    - An ask fill: you buy at ask, so inventory increases by 1 (you bought a unit)
    
    Wait, let me re-read: "A fill earns its depth and moves inventory one unit"
    - Bid hit: you are the maker, someone hits your bid, you sell, so your inventory (long position) decreases by 1
    - Ask hit: someone hits your ask, you buy, so your inventory increases by 1
    
    So:
    - bid_depth affects probability of selling (reducing q)
    - ask_depth affects probability of buying (increasing q)
    
    If q > 0 (long), we want to reduce q, so we want to increase bid fill probability (smaller bid_depth) and decrease ask fill probability (larger ask_depth).
    
    Wait, I had it backwards. Let me redo:
    - Smaller bid_depth → higher bid fill probability → more selling → q decreases
    - Larger bid_depth → lower bid fill probability → less selling → q stays higher
    
    So if q > 0 (long), we want to sell more, so we want SMALLER bid_depth and LARGER ask_depth.
    
    Therefore:
    - bid_depth = base_depth - alpha * q  (q > 0 → smaller bid_depth → more selling)
    - ask_depth = base_depth + alpha * q  (q > 0 → larger ask_depth → less buying)
    
    For q < 0 (short), we want to buy more:
    - bid_depth = base_depth - alpha * q  (q < 0 → larger bid_depth → less selling... wait that's wrong)
    
    Let me be more careful. If q < 0 (short, i.e., negative inventory means we owe units), we want to buy to reduce the short:
    - We want more ask fills (buying), so smaller ask_depth
    - We want fewer bid fills (selling), so larger bid_depth
    
    So:
    - bid_depth = base_depth - alpha * q
      - If q > 0: bid_depth = base - alpha*q < base (smaller, more selling) ✓
      - If q < 0: bid_depth = base - alpha*q > base (larger, less selling) ✓
    - ask_depth = base_depth + alpha * q
      - If q > 0: ask_depth = base + alpha*q > base (larger, less buying) ✓
      - If q < 0: ask_depth = base + alpha*q < base (smaller, more buying) ✓
    
    Great, so the formula is:
    bid_depth = base_depth - alpha * q
    ask_depth = base_depth + alpha * q
    
    Now for alpha: it should be proportional to the marginal inventory cost.
    The marginal cost of inventory is gamma * sigma^2 * q (derivative of (gamma/2)*sigma^2*q^2).
    
    A reasonable heuristic: alpha = gamma * sigma^2 * (T - t) / (2 * k * base_depth) or something similar.
    
    Let's think about units. The fill probability is 1 - exp(-A * exp(-k * depth) * dt).
    For small depth changes, the change in fill probability is approximately:
    dP/ddepth ≈ -A * exp(-k*depth) * k * dt
    
    At depth = 1/k: exp(-k*(1/k)) = exp(-1) = 1/e
    So dP/ddepth ≈ -A * (1/e) * k * dt
    
    The marginal benefit of adjusting depth to reduce inventory should balance the marginal cost.
    
    Let's use a simpler approach: set alpha proportional to gamma * sigma^2 / k, scaled by time remaining.
    
    Actually, let's just use:
    alpha = gamma * sigma^2 * (T - t) / k
    
    And clamp depths to be in a valid range: [-3/k, inf). But we want positive depths for meaningful quotes.
    Let's clamp to [min_depth, max_depth] where min_depth is a small positive number and max_depth is something reasonable.
    
    Actually, the constraint says values below -3/k are invalid. So depth can be as low as -3/k + epsilon.
    But negative depth doesn't make physical sense for a quote. Let's just ensure depth > -3/k.
    
    For safety, let's clamp to [max(-3/k + 0.01, 0.001), some_max].
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    
    q = inventory.astype(np.float64)
    remaining = T - t
    
    # Base depth: risk-neutral is 1/k
    base_depth = 1.0 / k
    
    # Hedging coefficient: proportional to inventory cost and time remaining
    # alpha has units of depth per unit inventory
    alpha = gamma * sigma * sigma * remaining / k
    
    # Compute depths
    bid_depths = base_depth - alpha * q
    ask_depths = base_depth + alpha * q
    
    # Clamp to valid range: must be > -3/k
    min_depth = -3.0 / k + 1e-6
    # Also, for practical purposes, let's not go too extreme
    max_depth = 10.0 / k  # reasonable upper bound
    
    bid_depths = np.clip(bid_depths, min_depth, max_depth)
    ask_depths = np.clip(ask_depths, min_depth, max_depth)
    
    # Handle boundary conditions: no bid fill at q=Q, no ask fill at q=-Q
    # At q=Q, we shouldn't quote bid (set to inf)
    # At q=-Q, we shouldn't quote ask (set to inf)
    Q = params["Q"]
    bid_depths[q >= Q] = np.inf
    ask_depths[q <= -Q] = np.inf
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
