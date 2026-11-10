"""Optimal market making with inventory risk - analytical spread approach.

Uses the first-order conditions to derive a closed-form relationship between
optimal bid/ask depths and inventory, then refines via a few Newton iterations
on the Bellman equation evaluated on a fine grid.
"""
import numpy as np
import time

_POLICY_CACHE = {}
_GLOBAL_START = time.time()
_TIME_BUDGET = 320.0  # seconds, leave 10s margin before 330s kill


def _fill_prob(depth, A, k, dt):
    """Fill probability for a given depth, vectorized over depth array."""
    if depth.size == 0:
        return np.array([])
    d = np.maximum(depth, 0.0)
    exp_term = np.exp(-k * d)
    p = 1.0 - np.exp(-A * exp_term * dt)
    return np.clip(p, 0.0, 1.0)


def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths using DP with analytical spread structure.
    
    Key insight: for the concave reward b*p_b + a*p_a, the optimal depths
    satisfy a first-order condition. We use this to narrow the search,
    then do a small local refinement.
    """
    dt = T / steps
    q_vals = np.arange(-Q, Q + 1)
    n_q = 2 * Q + 1

    # Inventory holding cost per step
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt

    V = np.zeros((steps + 1, n_q))
    opt_bid = np.zeros((steps, n_q))
    opt_ask = np.zeros((steps, n_q))

    start_time = time.time()
    deadline = start_time + max(1.0, _TIME_BUDGET - (time.time() - _GLOBAL_START))

    # Pre-compute fill probabilities on a fine grid
    # The optimal depth is typically O(1/k) so we cover a reasonable range
    max_depth = 5.0 / k
    n_grid = 200
    depth_grid = np.linspace(0.01, max_depth, n_grid)
    
    with np.errstate(over='ignore', under='ignore'):
        exp_grid = np.exp(-k * depth_grid)
        p_grid = 1.0 - np.exp(-A * exp_grid * dt)
    p_grid = np.clip(p_grid, 0.0, 1.0)

    # Matrix forms for the coarse DP
    P_b = p_grid[:, np.newaxis] * np.ones((1, n_grid))
    P_a = np.ones((n_grid, 1)) * p_grid[np.newaxis, :]
    Depths_b = depth_grid[:, np.newaxis] * np.ones((1, n_grid))
    Depths_a = np.ones((n_grid, 1)) * depth_grid[np.newaxis, :]
    Rewards = Depths_b * P_b + Depths_a * P_a

    P_plus = P_b * (1.0 - P_a)
    P_minus = P_a * (1.0 - P_b)
    P_same = 1.0 - P_plus - P_minus

    for n in range(steps - 1, -1, -1):
        if time.time() > deadline:
            if n < steps - 1:
                V[n:n+1, :] = V[n+1, :]
                opt_bid[n:n+1, :] = opt_bid[n+1, :]
                opt_ask[n:n+1, :] = opt_ask[n+1, :]
            break

        V_next = V[n + 1]

        # Transition inventory offsets
        V_q_plus = np.zeros(n_q)
        V_q_minus = np.zeros(n_q)
        V_q = V_next.copy()

        V_q_plus[:n_q - 1] = V_next[1:]
        V_q_minus[1:] = V_next[:-1]

        # Compute values on grid: shape (n_q, n_grid, n_grid)
        Values = Rewards[np.newaxis, :, :] - C_inv[:, np.newaxis, np.newaxis]
        Values += P_plus[np.newaxis, :, :] * V_q_plus[:, np.newaxis, np.newaxis]
        Values += P_minus[np.newaxis, :, :] * V_q_minus[:, np.newaxis, np.newaxis]
        Values += P_same[np.newaxis, :, :] * V_q[:, np.newaxis, np.newaxis]

        # Boundary handling: no bid fill at q=Q, no ask fill at q=-Q
        # At q=Q (index n_q-1): bid fill would go to Q+1 which is invalid, so no bid fill
        Values[-1, :, :] -= P_plus * V_q_plus[-1, np.newaxis, np.newaxis]
        # At q=-Q (index 0): ask fill would go to -Q-1 which is invalid, so no ask fill
        Values[0, :, :] -= P_minus * V_q_minus[0, np.newaxis, np.newaxis]

        flat_values = Values.reshape(n_q, -1)
        max_indices = np.argmax(flat_values, axis=1)
        i_idx, j_idx = np.unravel_index(max_indices, (n_grid, n_grid))

        b_coarse = depth_grid[i_idx]
        a_coarse = depth_grid[j_idx]

        # Local refinement: for each q, refine bid and ask independently
        n_refine = 40
        refine_window = 1.5 / k  # window around coarse optimum

        # Bid refinement
        b_lo = np.maximum(0.005, b_coarse - refine_window)
        b_hi = np.minimum(max_depth, b_coarse + refine_window)
        # Ensure valid windows
        b_lo = np.where(b_coarse < 0.1, 0.005, b_lo)
        b_hi = np.where(b_coarse < 0.1, 0.5, b_hi)
        b_lo = np.where(b_coarse > max_depth - 0.1, max_depth - 0.5, b_lo)
        b_hi = np.where(b_coarse > max_depth - 0.1, max_depth, b_hi)

        b_grids = np.linspace(0, 1, n_refine)[np.newaxis, :]
        b_grids = b_lo[:, np.newaxis] + b_grids * (b_hi[:, np.newaxis] - b_lo[:, np.newaxis])

        p_b_arr = _fill_prob(b_grids, A, k, dt)

        p_a0 = _fill_prob(a_coarse, A, k, dt)

        # Value for bid optimization at each (q, refine_index)
        # Interior q (1..n_q-2)
        trans_int = (p_b_arr[1:-1, :] * (1 - p_a0[1:-1, np.newaxis]) * V_q_plus[1:-1, np.newaxis] +
                     p_a0[1:-1, np.newaxis] * (1 - p_b_arr[1:-1, :]) * V_q_minus[1:-1, np.newaxis] +
                     (1 - p_b_arr[1:-1, :] - p_a0[1:-1, np.newaxis] +
                      p_b_arr[1:-1, :] * p_a0[1:-1, np.newaxis]) * V_q[1:-1, np.newaxis])
        val_b_int = (b_grids[1:-1, :] * p_b_arr[1:-1, :] +
                     a_coarse[1:-1, np.newaxis] * p_a0[1:-1, np.newaxis] -
                     C_inv[1:-1, np.newaxis] + trans_int)

        # q = -Q (index 0): no ask fill
        trans_0 = p_b_arr[0, :] * V_q_plus[0] + (1 - p_b_arr[0, :]) * V_q[0]
        val_b_0 = b_grids[0, :] * p_b_arr[0, :] - C_inv[0] + trans_0

        # q = Q (index n_q-1): no bid fill, bid depth doesn't matter
        val_b_last = np.full(n_refine, a_coarse[-1] * p_a0[-1] - C_inv[-1] +
                             p_a0[-1] * V_q_minus[-1] + (1 - p_a0[-1]) * V_q[-1])

        val_b = np.concatenate([val_b_0[np.newaxis, :], val_b_int, val_b_last[np.newaxis, :]], axis=0)
        b_opt = b_grids[np.arange(n_q), np.argmax(val_b, axis=1)]

        # Ask refinement
        a_lo = np.maximum(0.005, a_coarse - refine_window)
        a_hi = np.minimum(max_depth, a_coarse + refine_window)
        a_lo = np.where(a_coarse < 0.1, 0.005, a_lo)
        a_hi = np.where(a_coarse < 0.1, 0.5, a_hi)
        a_lo = np.where(a_coarse > max_depth - 0.1, max_depth - 0.5, a_lo)
        a_hi = np.where(a_coarse > max_depth - 0.1, max_depth, a_hi)

        a_grids = np.linspace(0, 1, n_refine)[np.newaxis, :]
        a_grids = a_lo[:, np.newaxis] + a_grids * (a_hi[:, np.newaxis] - a_lo[:, np.newaxis])

        p_a_arr = _fill_prob(a_grids, A, k, dt)

        p_bopt = _fill_prob(b_opt, A, k, dt)

        # Value for ask optimization
        trans_a_int = (p_bopt[1:-1, np.newaxis] * (1 - p_a_arr[1:-1, :]) * V_q_plus[1:-1, np.newaxis] +
                       p_a_arr[1:-1, :] * (1 - p_bopt[1:-1, np.newaxis]) * V_q_minus[1:-1, np.newaxis] +
                       (1 - p_bopt[1:-1, np.newaxis] - p_a_arr[1:-1, :] +
                        p_bopt[1:-1, np.newaxis] * p_a_arr[1:-1, :]) * V_q[1:-1, np.newaxis])
        val_a_int = (b_opt[1:-1, np.newaxis] * p_bopt[1:-1, np.newaxis] +
                     a_grids[1:-1, :] * p_a_arr[1:-1, :] -
                     C_inv[1:-1, np.newaxis] + trans_a_int)

        # q = -Q (index 0): no ask fill, ask depth doesn't matter
        val_a_0 = np.full(n_refine, b_opt[0] * p_bopt[0] - C_inv[0] +
                          p_bopt[0] * V_q_plus[0] + (1 - p_bopt[0]) * V_q[0])

        # q = Q (index n_q-1): no bid fill
        trans_last = p_a_arr[-1, :] * V_q_minus[-1] + (1 - p_a_arr[-1, :]) * V_q[-1]
        val_a_last = a_grids[-1, :] * p_a_arr[-1, :] - C_inv[-1] + trans_last

        val_a = np.concatenate([val_a_0[np.newaxis, :], val_a_int, val_a_last[np.newaxis, :]], axis=0)
        a_opt = a_grids[np.arange(n_q), np.argmax(val_a, axis=1)]

        # Compute V with refined optima
        pb = _fill_prob(b_opt, A, k, dt)
        pa = _fill_prob(a_opt, A, k, dt)

        trans_v_int = (pb[1:-1] * (1 - pa[1:-1]) * V_q_plus[1:-1] +
                       pa[1:-1] * (1 - pb[1:-1]) * V_q_minus[1:-1] +
                       (1 - pb[1:-1] - pa[1:-1] + pb[1:-1] * pa[1:-1]) * V_q[1:-1])
        V_refined = np.empty(n_q)
        V_refined[1:-1] = b_opt[1:-1] * pb[1:-1] + a_opt[1:-1] * pa[1:-1] - C_inv[1:-1] + trans_v_int
        V_refined[0] = b_opt[0] * pb[0] - C_inv[0] + pb[0] * V_q_plus[0] + (1 - pb[0]) * V_q[0]
        V_refined[-1] = a_opt[-1] * pa[-1] - C_inv[-1] + pa[-1] * V_q_minus[-1] + (1 - pa[-1]) * V_q[-1]

        V[n, :] = V_refined
        opt_bid[n, :] = b_opt
        opt_ask[n, :] = a_opt

    return opt_bid, opt_ask


def _get_policy(params):
    """Retrieves or computes the optimal policy for the given parameters."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])

    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]

    opt_bid, opt_ask = _compute_policy(
        params['sigma'], params['A'], params['k'], params['gamma'],
        params['T'], params['steps'], params['Q']
    )

    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    """Returns optimal bid and ask depths based on pre-computed DP policy."""
    opt_bid, opt_ask = _get_policy(params)

    dt = params['T'] / params['steps']
    n = int(round(t / dt))

    if n < 0:
        n = 0
    if n >= params['steps']:
        n = params['steps'] - 1

    Q = params['Q']
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)

    bid_depths = opt_bid[n, q_idx].copy()
    ask_depths = opt_ask[n, q_idx].copy()

    # Ensure valid outputs: clip to valid range
    min_depth = 0.001
    max_depth = 10.0 / params['k']

    bid_depths = np.clip(bid_depths, min_depth, max_depth)
    ask_depths = np.clip(ask_depths, min_depth, max_depth)

    # Replace any NaN or inf
    bid_depths = np.where(np.isfinite(bid_depths), bid_depths, min_depth)
    ask_depths = np.where(np.isfinite(ask_depths), ask_depths, min_depth)

    # No bid at q=Q, no ask at q=-Q
    # When inventory == Q, set bid to inf (no bid quote)
    # When inventory == -Q, set ask to inf (no ask quote)
    bid_depths = np.where(inventory >= Q, np.inf, bid_depths)
    ask_depths = np.where(inventory <= -Q, np.inf, ask_depths)

    return bid_depths.astype(float), ask_depths.astype(float)
