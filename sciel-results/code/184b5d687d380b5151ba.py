"""
Market making with inventory risk: pre-computed DP policy with high-resolution grid.

Structural idea: solve the finite-horizon MDP exactly via backward induction on a
fine depth grid, caching results per regime. The depth grid is log-spaced with
high resolution around the expected optimal region (~1/k) and extends to capture
boundary behavior. A polish pass refines the grid locally around initially-found
optima for higher precision.
"""
import numpy as np
import time

_policy_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 320.0  # safety margin before 330s kill


def _heuristic_depths(params, t, inventory):
    """Analytical fallback if DP doesn't finish in time."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    Q = params['Q']
    q = np.asarray(inventory, dtype=np.float64)
    t_remaining = np.maximum(T - t, 0.0)
    d0 = 1.0 / k
    skew_scale = gamma * sigma * sigma * t_remaining
    alpha = skew_scale / (A * k) if A > 0 and k > 0 else 0.0
    alpha = min(alpha, 6.0 / k)
    bid_depth = d0 + alpha * q
    ask_depth = d0 - alpha * q
    min_depth = 0.01 / k
    max_depth = 10.0 / k
    bid_depth = np.clip(bid_depth, min_depth, max_depth)
    ask_depth = np.clip(ask_depth, min_depth, max_depth)
    bid_depth = np.where(inventory >= Q, np.inf, bid_depth)
    ask_depth = np.where(inventory <= -Q, np.inf, ask_depth)
    bid_depth = np.where(np.isfinite(bid_depth) & (bid_depth >= -3.0 / k), bid_depth, np.inf)
    ask_depth = np.where(np.isfinite(ask_depth) & (ask_depth >= -3.0 / k), ask_depth, np.inf)
    return bid_depth, ask_depth


def _compute_optimal_policy(params):
    """
    Compute optimal bid/ask depths for all (step, inventory) states via backward DP.
    Uses a log-spaced depth grid with high resolution near 1/k.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    n_q = 2 * Q + 1
    q_range = np.arange(-Q, Q + 1)

    # Depth grid: log-spaced, high resolution near 1/k
    depth_min = 0.01
    depth_max = 10.0 / k
    n_inner = 150
    n_outer = 50
    inner_max = 3.0 / k
    inner_grid = np.geomspace(depth_min, inner_max, n_inner)
    outer_grid = np.geomspace(inner_max, depth_max, n_outer)
    depth_grid = np.concatenate([inner_grid, outer_grid])
    depth_grid = np.unique(depth_grid)
    n_depths = len(depth_grid)

    # Precompute fill probabilities for all depths
    exp_term = np.exp(-k * depth_grid)
    p_fill = 1.0 - np.exp(-A * exp_term * dt)

    V = np.zeros((steps + 1, n_q))
    bid_table = np.zeros((steps, n_q))
    ask_table = np.zeros((steps, n_q))

    inv_cost_factor = (gamma / 2.0) * sigma**2 * dt

    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 5:
            for n2 in range(n, -1, -1):
                t2 = n2 * dt
                for qi in range(n_q):
                    q = float(q_range[qi])
                    d = 1.0 / k + gamma * sigma**2 * q * (T - t2) / (A * k)
                    d = max(d, depth_min)
                    bid_table[n2, qi] = d
                    ask_table[n2, qi] = d
            break

        Vn1 = V[n + 1]

        for qi in range(n_q):
            q = float(q_range[qi])
            cost = inv_cost_factor * q * q

            Vn1_q = Vn1[qi]

            if q == Q:
                p_bid_eff = np.zeros(n_depths)
            else:
                p_bid_eff = p_fill

            if q == -Q:
                p_ask_eff = np.zeros(n_depths)
            else:
                p_ask_eff = p_fill

            if q < Q:
                Vn1_q_plus = Vn1[qi + 1]
            else:
                Vn1_q_plus = Vn1_q

            if q > -Q:
                Vn1_q_minus = Vn1[qi - 1]
            else:
                Vn1_q_minus = Vn1_q

            dV_plus = Vn1_q_plus - Vn1_q
            dV_minus = Vn1_q_minus - Vn1_q
            cross_term = 2.0 * Vn1_q - Vn1_q_plus - Vn1_q_minus

            bid_contrib = p_bid_eff * (depth_grid + dV_plus)
            ask_contrib = p_ask_eff * (depth_grid + dV_minus)

            M = (bid_contrib[:, None] + ask_contrib[None, :]
                 + cross_term * np.outer(p_bid_eff, p_ask_eff)
                 + Vn1_q - cost)

            idx_max = np.argmax(M, axis=1)
            best_val_i = M[np.arange(n_depths), idx_max]

            best_i = np.argmax(best_val_i)
            best_val = best_val_i[best_i]
            best_bid = depth_grid[best_i]
            best_ask = depth_grid[idx_max[best_i]]

            V[n, qi] = best_val
            bid_table[n, qi] = best_bid
            ask_table[n, qi] = best_ask

    # Multi-pass polish: progressively tighter windows for sub-grid accuracy
    if time.time() - _START_TIME < TIME_LIMIT - 10:
        bid_table, ask_table = _polish_policy(
            params, bid_table, ask_table, depth_grid, p_fill, V, q_range,
            n_q, dt, k, A, sigma, gamma, T, steps, Q,
            local_half_width=0.05, n_local=40
        )
    if time.time() - _START_TIME < TIME_LIMIT - 10:
        bid_table, ask_table = _polish_policy(
            params, bid_table, ask_table, depth_grid, p_fill, V, q_range,
            n_q, dt, k, A, sigma, gamma, T, steps, Q,
            local_half_width=0.033, n_local=40
        )

    return bid_table, ask_table


def _polish_policy(params, bid_table, ask_table, depth_grid, p_fill, V, q_range,
                   n_q, dt, k, A, sigma, gamma, T, steps, Q,
                   local_half_width=0.05, n_local=40):
    """
    High-precision polish: for each (step, inventory), create a local fine grid
    around the initially-found optimal bid and ask depths, recompute, and update.
    """
    bid_refined = bid_table.copy()
    ask_refined = ask_table.copy()
    V_refined = V.copy()

    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 5:
            break

        inv_cost_factor = (gamma / 2.0) * sigma**2 * dt
        Vn1 = V_refined[n + 1]

        for qi in range(n_q):
            q = float(q_range[qi])
            cost = inv_cost_factor * q * q
            Vn1_q = Vn1[qi]

            if q == Q:
                p_bid_base = np.zeros(1)
            else:
                p_bid_base = p_fill

            if q == -Q:
                p_ask_base = np.zeros(1)
            else:
                p_ask_base = p_fill

            if q < Q:
                Vn1_q_plus = Vn1[qi + 1]
            else:
                Vn1_q_plus = Vn1_q

            if q > -Q:
                Vn1_q_minus = Vn1[qi - 1]
            else:
                Vn1_q_minus = Vn1_q

            dV_plus = Vn1_q_plus - Vn1_q
            dV_minus = Vn1_q_minus - Vn1_q
            cross_term = 2.0 * Vn1_q - Vn1_q_plus - Vn1_q_minus

            best_bid = bid_table[n, qi]
            best_ask = ask_table[n, qi]

            if np.isinf(best_bid):
                bid_local = depth_grid[:n_local]
            else:
                lo = max(0.005, best_bid - local_half_width / k)
                hi = best_bid + local_half_width / k
                bid_local = np.geomspace(lo, hi, n_local) if lo > 0 else np.linspace(0.005, hi, n_local)

            if np.isinf(best_ask):
                ask_local = depth_grid[:n_local]
            else:
                lo = max(0.005, best_ask - local_half_width / k)
                hi = best_ask + local_half_width / k
                ask_local = np.geomspace(lo, hi, n_local) if lo > 0 else np.linspace(0.005, hi, n_local)

            p_bid_local = 1.0 - np.exp(-A * np.exp(-k * bid_local) * dt)
            p_ask_local = 1.0 - np.exp(-A * np.exp(-k * ask_local) * dt)

            if q == Q:
                p_bid_local = np.zeros_like(p_bid_local)
            if q == -Q:
                p_ask_local = np.zeros_like(p_ask_local)

            bid_contrib = p_bid_local * (bid_local + dV_plus)
            ask_contrib = p_ask_local * (ask_local + dV_minus)

            M = (bid_contrib[:, None] + ask_contrib[None, :]
                 + cross_term * np.outer(p_bid_local, p_ask_local)
                 + Vn1_q - cost)

            idx_max = np.argmax(M, axis=1)
            best_val_i = M[np.arange(len(bid_local)), idx_max]
            best_i = np.argmax(best_val_i)
            best_val = best_val_i[best_i]

            V_refined[n, qi] = best_val
            bid_refined[n, qi] = bid_local[best_i]
            ask_refined[n, qi] = ask_local[idx_max[best_i]]

    return bid_refined, ask_refined


def quote(t, inventory, mid, params):
    """
    Return (bid_depths, ask_depths) arrays of shape (N,) from pre-computed DP policy.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    cache_key = (sigma, A, k, gamma, T, steps, Q)

    if cache_key not in _policy_cache:
        bid_table, ask_table = _compute_optimal_policy(params)
        _policy_cache[cache_key] = (bid_table, ask_table)
    else:
        bid_table, ask_table = _policy_cache[cache_key]

    n = int(round(t / (T / steps)))
    n = np.clip(n, 0, steps - 1)

    q = np.asarray(inventory, dtype=np.int64)
    idx = q + Q
    idx = np.clip(idx, 0, 2 * Q)

    bid_depths = bid_table[n, idx].astype(np.float64)
    ask_depths = ask_table[n, idx].astype(np.float64)

    bid_depths[q >= Q] = np.inf
    ask_depths[q <= -Q] = np.inf

    bid_depths[np.isnan(bid_depths)] = np.inf
    ask_depths[np.isnan(ask_depths)] = np.inf
    bid_depths[bid_depths < -3.0 / k] = np.inf
    ask_depths[ask_depths < -3.0 / k] = np.inf
    bid_depths[bid_depths == -np.inf] = np.inf
    ask_depths[ask_depths == -np.inf] = np.inf

    return bid_depths, ask_depths
