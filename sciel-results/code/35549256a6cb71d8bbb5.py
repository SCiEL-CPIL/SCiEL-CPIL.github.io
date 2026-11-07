"""
Optimal market making with inventory risk.
Uses piecewise-linear value function approximation in inventory.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for precomputed DP tables keyed by (sigma, A, k, gamma, T, steps, Q)
_dp_cache = {}


def _log1p_neg(x):
    """Compute log(1 - x) safely for x in [0, 1)."""
    x = np.clip(x, 0.0, 1.0 - 1e-15)
    return np.log1p(-x)


def _compute_dp_table(sigma, A, k, gamma, T, steps, Q):
    """
    Precompute the optimal bid/ask depths for each step and inventory level.
    Returns arrays bid_depths_table[steps, 2*Q+1] and ask_depths_table[steps, 2*Q+1].
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _dp_cache:
        return _dp_cache[key]

    dt = T / steps
    n_inv = 2 * Q + 1
    inv_levels = np.arange(-Q, Q + 1, dtype=np.float64)
    inv_idx = np.arange(n_inv)

    # Value function V[step, inv_idx]
    V = np.zeros((steps + 1, n_inv), dtype=np.float64)

    # For the last step, V[steps, :] = 0

    # We'll store optimal depths
    bid_depths_table = np.zeros((steps, n_inv), dtype=np.float64)
    ask_depths_table = np.zeros((steps, n_inv), dtype=np.float64)

    # Work backwards from step = steps-1 down to 0
    # V[s] depends on V[s+1]
    for s in range(steps - 1, -1, -1):
        # Inventory cost for this step
        inv_cost = 0.5 * gamma * sigma**2 * inv_levels**2 * dt

        # For each inventory level, we need to choose bid_depth and ask_depth
        # to maximize: -inv_cost + E[reward + V[s+1, new_inv]]
        #
        # The expected reward from bid side (if inventory < Q):
        #   p_bid = 1 - exp(-A * exp(-k * bid_depth) * dt)
        #   If bid fills: +bid_depth, inventory decreases by 1
        #   Expected bid contribution = p_bid * (bid_depth + V[s+1, inv-1]) + (1-p_bid) * V[s+1, inv]
        #   = p_bid * bid_depth + V[s+1, inv] + p_bid * (V[s+1, inv-1] - V[s+1, inv])
        #
        # Similarly for ask side (if inventory > -Q):
        #   p_ask = 1 - exp(-A * exp(-k * ask_depth) * dt)
        #   Expected ask contribution = p_ask * ask_depth + V[s+1, inv] + p_ask * (V[s+1, inv+1] - V[s+1, inv])
        #
        # Total expected value:
        #   V[s, inv] = -inv_cost + V[s+1, inv]
        #     + p_bid * bid_depth + p_bid * (V[s+1, inv-1] - V[s+1, inv])  [if inv < Q]
        #     + p_ask * ask_depth + p_ask * (V[s+1, inv+1] - V[s+1, inv])  [if inv > -Q]
        #
        # To maximize over bid_depth and ask_depth independently:
        #   Maximize: p_bid * bid_depth + p_bid * delta_V_bid
        #   where p_bid = 1 - exp(-A * exp(-k * d) * dt)
        #
        # Let x = A * exp(-k * d) * dt, so p = 1 - exp(-x)
        # d = -(1/k) * ln(x / (A * dt))
        #
        # Objective: (1 - exp(-x)) * (-(1/k) * ln(x / (A*dt))) + (1 - exp(-x)) * delta_V
        #
        # This is a 1D optimization in x > 0. We can solve it numerically.
        # But for speed, we can use a few iterations or a lookup.
        #
        # Actually, let's use a simpler approach: for each inventory level,
        # we can compute the optimal depth by solving the first-order condition
        # or by a small grid search. Given the structure, a few Newton steps
        # or a precomputed table over a range of delta_V values would work.
        #
        # For now, let's use a vectorized approach with a moderate grid.

        V_next = V[s + 1]  # shape (n_inv,)

        # Compute delta_V for bid: V[s+1, inv-1] - V[s+1, inv] for inv in [0, Q-1]
        # For inv = Q, no bid fill possible
        # Compute delta_V for ask: V[s+1, inv+1] - V[s+1, inv] for inv in [-Q+1, Q]
        # For inv = -Q, no ask fill possible

        # We'll iterate over inventory levels
        for i in range(n_inv):
            inv = inv_levels[i]
            dv = -inv_cost[i] + V_next[i]

            # Bid side
            if i > 0:  # inv < Q
                delta_bid = V_next[i - 1] - V_next[i]
                # Optimize over bid_depth
                # Use a simple grid + refinement
                bid_depths_table[s, i] = _optimal_depth(A, k, dt, delta_bid)
            else:
                bid_depths_table[s, i] = np.inf  # no bid

            # Ask side
            if i < n_inv - 1:  # inv > -Q
                delta_ask = V_next[i + 1] - V_next[i]
                ask_depths_table[s, i] = _optimal_depth(A, k, dt, delta_ask)
            else:
                ask_depths_table[s, i] = np.inf  # no ask

            V[s, i] = dv + (bid_depths_table[s, i] if np.isfinite(bid_depths_table[s, i]) else 0.0) * 0  # placeholder
            # Actually we need to compute the full expected value
            # Let's recompute properly

        # Recompute V[s] properly
        for i in range(n_inv):
            inv = inv_levels[i]
            dv_base = -inv_cost[i] + V_next[i]

            bid_d = bid_depths_table[s, i]
            ask_d = ask_depths_table[s, i]

            bid_contrib = 0.0
            ask_contrib = 0.0

            if np.isfinite(bid_d):
                p_bid = 1.0 - np.exp(-A * np.exp(-k * bid_d) * dt)
                delta_bid = V_next[i - 1] - V_next[i]
                bid_contrib = p_bid * (bid_d + delta_bid)

            if np.isfinite(ask_d):
                p_ask = 1.0 - np.exp(-A * np.exp(-k * ask_d) * dt)
                delta_ask = V_next[i + 1] - V_next[i]
                ask_contrib = p_ask * (ask_d + delta_ask)

            V[s, i] = dv_base + bid_contrib + ask_contrib

    _dp_cache[key] = (bid_depths_table, ask_depths_table)
    return _dp_cache[key]


def _optimal_depth(A, k, dt, delta_V):
    """
    Find optimal depth d >= 0 maximizing:
      f(d) = (1 - exp(-A exp(-k d) dt)) * (d + delta_V)
    for delta_V in R.
    """
    # If delta_V is very large negative, we want minimal quoting (d -> large)
    # If delta_V is very large positive, we want aggressive quoting (d -> small)
    
    # Use a simple approach: try a few candidate depths
    # The optimal d is where the marginal benefit equals marginal cost
    # For a quick approximation, we can use the fact that for small A*dt,
    # p ≈ A*exp(-k*d)*dt, so f ≈ A*exp(-k*d)*dt*(d + delta_V)
    # Maximizing this: derivative = A*dt*exp(-k*d)*(1 - k*(d + delta_V)) = 0
    # => d + delta_V = 1/k => d = 1/k - delta_V
    
    # But this is only valid for small p. Let's use a more robust method.
    
    # Grid search over a reasonable range of d
    # d is typically in [0, ~5/k]
    d_min = 0.0
    d_max = 10.0 / k if k > 0 else 10.0
    n_grid = 50
    d_vals = np.linspace(d_min, d_max, n_grid)
    
    p_vals = 1.0 - np.exp(-A * np.exp(-k * d_vals) * dt)
    f_vals = p_vals * (d_vals + delta_V)
    
    best_idx = np.argmax(f_vals)
    best_d = d_vals[best_idx]
    
    # Refine with a few steps of golden section or just return the grid result
    # For speed, return the grid result
    return best_d


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    # Get or compute DP table
    bid_depths_table, ask_depths_table = _compute_dp_table(sigma, A, k, gamma, T, steps, Q)

    # Current step
    step = int(round(t / (T / steps)))
    step = min(step, steps - 1)

    # Map inventory to index
    inv_idx = inventory + Q  # inventory in [-Q, Q] -> [0, 2Q]

    # Look up depths
    bid_depths = bid_depths_table[step, inv_idx]
    ask_depths = ask_depths_table[step, inv_idx]

    # Ensure valid output: no values below -3/k
    min_depth = -3.0 / k
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
