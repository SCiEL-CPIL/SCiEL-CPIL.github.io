"""Optimal market making with inventory risk — DP-based policy with vectorized depth search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}
_START_TIME = time.time()
_DEADLINE = 315.0  # Safety margin before 330s kill


def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths via DP over (time, inventory) states.
    Uses a non-uniform depth grid concentrated near the analytically-optimal region.
    Vectorized over inventory for speed.
    """
    dt = T / steps
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]

    # Depth grid: non-uniform, concentrated near 1/k where optimal depths typically lie
    depth_min = 1e-4
    depth_max = max(12.0 / k, 2.0)
    # Use log-spaced grid for better resolution at small depths
    n_depths = 120
    depth_grid = np.geomspace(depth_min, depth_max, n_depths)
    # Add a few uniform points to catch the peak
    uniform_pts = np.linspace(0.5 / k, 2.0 / k, 20)
    depth_grid = np.unique(np.concatenate([depth_grid, uniform_pts]))
    depth_grid = np.sort(depth_grid)
    n_depths = len(depth_grid)

    # Precompute fill probabilities for each depth
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)

    # V_next[q_idx] where q_idx = q + Q, q in [-Q, Q]
    V_next = np.zeros(2 * Q + 1)

    # Store optimal depths: shape (steps, 2*Q+1)
    opt_bid = np.full((steps, 2 * Q + 1), np.inf)
    opt_ask = np.full((steps, 2 * Q + 1), np.inf)

    q_vals = np.arange(-Q, Q + 1)
    penalties = (gamma / 2.0) * sigma**2 * q_vals**2 * dt

    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 3.0:
            # Fill remaining steps with current best estimate
            for nn in range(n - 1, -1, -1):
                opt_bid[nn] = opt_bid[n]
                opt_ask[nn] = opt_ask[n]
            break

        V_curr = np.zeros(2 * Q + 1)

        # Vectorized optimization for interior states (q in (-Q, Q))
        interior = np.arange(1, 2 * Q)  # indices 1..2Q-1 correspond to q=-Q+1..Q-1

        delta_plus = V_next[interior + 1] - V_next[interior]      # V[q+1] - V[q]
        delta_minus = V_next[interior - 1] - V_next[interior]     # V[q-1] - V[q]

        # For bid: maximize p(d) * (d + delta_plus) over d
        # For ask: maximize p(d) * (d + delta_minus) over d
        # Shape: (n_interior, n_depths)
        bid_scores = fill_probs[np.newaxis, :] * (depth_grid[np.newaxis, :] + delta_plus[:, np.newaxis])
        ask_scores = fill_probs[np.newaxis, :] * (depth_grid[np.newaxis, :] + delta_minus[:, np.newaxis])

        best_bid_idx = np.argmax(bid_scores, axis=1)
        best_ask_idx = np.argmax(ask_scores, axis=1)

        best_bid_val = bid_scores[np.arange(len(interior)), best_bid_idx]
        best_ask_val = ask_scores[np.arange(len(interior)), best_ask_idx]

        V_curr[interior] = best_bid_val + best_ask_val - penalties[interior] + V_next[interior]
        opt_bid[n, interior] = depth_grid[best_bid_idx]
        opt_ask[n, interior] = depth_grid[best_ask_idx]

        # Boundary: q = Q (only ask)
        q_idx = 2 * Q
        delta = V_next[q_idx - 1] - V_next[q_idx]
        vals = fill_probs * (depth_grid + delta)
        best_i = np.argmax(vals)
        V_curr[q_idx] = vals[best_i] - penalties[q_idx] + V_next[q_idx]
        opt_ask[n, q_idx] = depth_grid[best_i]
        opt_bid[n, q_idx] = np.inf

        # Boundary: q = -Q (only bid)
        q_idx = 0
        delta = V_next[q_idx + 1] - V_next[q_idx]
        vals = fill_probs * (depth_grid + delta)
        best_i = np.argmax(vals)
        V_curr[q_idx] = vals[best_i] - penalties[q_idx] + V_next[q_idx]
        opt_bid[n, q_idx] = depth_grid[best_i]
        opt_ask[n, q_idx] = np.inf

        V_next = V_curr

    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return _POLICY_CACHE[key]


def _analytical_quote(t, inventory, mid, params):
    """
    Avellaneda-Stoikov style analytical quote as fallback.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    dt = T / steps
    remaining = T - t

    base = 1.0 / k
    skew_coeff = gamma * sigma**2 * remaining / 2.0
    c = skew_coeff / (2.0 * k)

    bid_depths = base + c * inventory
    ask_depths = base - c * inventory

    bid_depths = np.clip(bid_depths, 1e-4, 12.0 / k)
    ask_depths = np.clip(ask_depths, 1e-4, 12.0 / k)

    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """Quote using pre-computed optimal policy with analytical fallback."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    n = int(round(t / (T / steps)))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0

    key = (sigma, A, k, gamma, T, steps, Q)

    if key not in _POLICY_CACHE:
        if time.time() < _DEADLINE - 15.0:
            try:
                _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q)
            except Exception:
                pass

    if key in _POLICY_CACHE:
        opt_bid, opt_ask = _POLICY_CACHE[key]
        q_idx = np.clip(inventory + Q, 0, 2 * Q)
        bid_depths = opt_bid[n, q_idx].copy()
        ask_depths = opt_ask[n, q_idx].copy()
        bid_depths = np.where(bid_depths < 1e-4, np.inf, bid_depths)
        ask_depths = np.where(ask_depths < 1e-4, np.inf, ask_depths)
        bid_depths = np.asarray(bid_depths, dtype=np.float64)
        ask_depths = np.asarray(ask_depths, dtype=np.float64)
        return bid_depths, ask_depths
    else:
        return _analytical_quote(t, inventory, mid, params)
# EVOLVE-BLOCK-END
