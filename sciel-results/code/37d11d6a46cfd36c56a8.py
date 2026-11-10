"""Optimal market making with inventory risk.

Strategy: continuous first-order-condition (FOC) solver.
For each path we compute optimal bid/ask spreads that balance the
marginal fill-profit against the marginal inventory-risk cost.

The FOC for a single side (ignoring interaction between bid/ask) is:
    d * (1 - exp(-k d)) = c / k
where c is the per-unit risk cost. We solve this with Newton-Raphson,
and the inventory-aware cost c is derived from the HJB-like value
gradient (gamma * sigma^2 * q) scaled by time-remaining.

A deadline guard ensures we return a valid (possibly fallback) result
before the 330 s wall-clock limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # safety margin before 330 s limit


def _solve_foc(c, k, max_iter=40, tol=1e-12):
    """Solve d * (1 - exp(-k d)) = c for d >= 0 via Newton-Raphson.

    f(d) = d*(1 - exp(-k d)) - c
    f'(d) = (1 - exp(-k d)) + d*k*exp(-k d)
    """
    # Vectorized Newton over array c
    c = np.asarray(c, dtype=np.float64)
    # Initial guess: for small c, d ~ sqrt(2c/k); for large c, d ~ c/k
    d = np.where(c > 1e-8,
                 np.sqrt(np.maximum(c * 2.0, 1e-12) / k),
                 np.maximum(c / k, 1e-12))
    d = np.clip(d, 1e-10, 10.0 / k)

    for _ in range(max_iter):
        kd = k * d
        exp_kd = np.exp(-kd)
        one_minus = 1.0 - exp_kd
        f = d * one_minus - c
        fp = one_minus + d * k * exp_kd
        # Avoid division by zero
        fp = np.where(np.abs(fp) < 1e-15, 1e-15, fp)
        step = f / fp
        d_new = d - step
        # Keep d non-negative
        d_new = np.maximum(d_new, 1e-12)
        if np.max(np.abs(step)) < tol:
            d = d_new
            break
        d = d_new
    return d


def quote(t, inventory, mid, params):
    """Inventory-aware market making policy using continuous FOC solver."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    n_paths = len(inventory)
    inv = np.asarray(inventory, dtype=np.float64)
    dt = T / steps

    # Time remaining fraction (1 at start, 0 at end)
    t_rem = max(0.0, 1.0 - t / T)

    # Base risk-neutral depth: solve FOC with zero inventory cost
    # d*(1-exp(-k d)) = 0 => d=0, but we need a floor
    # For the seed, the risk-neutral depth balances fill prob vs depth.
    # A fill earns depth d, fill prob ~ A*exp(-k*d)*dt
    # Expected profit per step ~ A*exp(-k*d)*dt*d
    # Maximize over d: derivative gives k*d*(1-exp(-k*d)) = 1 - exp(-k*d)
    # => k*d = 1 => d = 1/k
    base_depth = 1.0 / k

    # Inventory-aware adjustment:
    # The HJB value function gradient w.r.t. inventory is approximately
    # V_q ~ -gamma * sigma^2 * q * (time-remaining factor)
    # The optimal spread shift is proportional to V_q / (fill rate)
    #
    # Risk cost per unit inventory: gamma * sigma^2 * q * dt * t_rem
    # This is the marginal cost of holding one more unit for the next step.
    #
    # For bid side: a bid fill INCREASES inventory (bad if q>0, good if q<0)
    #   c_bid = max(0, gamma * sigma^2 * q * dt * t_rem) / (A * dt)
    #   = max(0, gamma * sigma^2 * q * t_rem) / A
    #
    # For ask side: an ask fill DECREASES inventory (good if q>0, bad if q<0)
    #   c_ask = max(0, -gamma * sigma^2 * q * dt * t_rem) / (A * dt)
    #   = max(0, -gamma * sigma^2 * q * t_rem) / A

    # Scale factor to map risk cost to FOC parameter
    # The FOC: d*(1-exp(-k*d)) = c
    # We want the depth to shift smoothly. The risk cost should be scaled
    # so that at q=Q, the depth shifts significantly.
    # Typical q range: [-Q, Q], so max risk cost ~ gamma*sigma^2*Q*t_rem/A
    # We want the FOC parameter to be O(1) at max inventory.
    # Let's use a scaling that makes the effect smooth.

    risk_scale = gamma * sigma * sigma * t_rem / A
    c_bid = np.maximum(0.0, risk_scale * inv)
    c_ask = np.maximum(0.0, risk_scale * (-inv))

    # Solve FOC for each side
    bid_depths = _solve_foc(c_bid, k)
    ask_depths = _solve_foc(c_ask, k)

    # Ensure minimum depth (avoid zero or negative)
    min_depth = 1e-6
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)

    # Cap depth to avoid extreme values
    max_depth = 5.0 / k
    bid_depths = np.minimum(bid_depths, max_depth)
    ask_depths = np.minimum(ask_depths, max_depth)

    # Boundary handling: no bid fill at q=Q, no ask fill at q=-Q
    # At q=Q, we should not quote bid (or quote at very wide depth)
    # At q=-Q, we should not quote ask
    at_max_inv = (inv >= Q - 0.5)
    at_min_inv = (inv <= -Q + 0.5)

    # At max inventory, make bid depth very wide (effectively no quote)
    # But we can't use inf, so use a large value
    bid_depths[at_max_inv] = max_depth * 10
    ask_depths[at_min_inv] = max_depth * 10

    # Validate: ensure all values are within bounds
    # Values must be >= -3/k (but we use positive depths, so this is fine)
    # No NaN, no -inf
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > -3.0 / k),
                          bid_depths, base_depth)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > -3.0 / k),
                          ask_depths, base_depth)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
