"""
Optimal market making with inventory risk.
Combines vectorized DP with per-state ternary-search polish.
Uses the vectorized grid DP for fast initialization, then refines each
state's depths with alternating ternary search on the exact continuous
objective for bid/ask depths.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 320.0  # safety margin before 330s kill


def _hmm_depths(params, t, inventory):
    """
    Hopenhaver-Sussmann closed-form approximation (fallback).
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    Q = params['Q']

    q = inventory.astype(np.float64)
    tau = np.maximum(T - t, 0.0)

    base = 1.0 / k
    skew = (gamma * sigma * sigma * q * tau) / (A * k) if A > 0 and k > 0 else np.zeros_like(q)

    bid_depth = base + skew
    ask_depth = base - skew

    min_depth = 1e-6
    max_depth = 15.0 / k
    bid_depth = np.clip(bid_depth, min_depth, max_depth)
    ask_depth = np.clip(ask_depth, min_depth, max_depth)

    bid_depth = np.where(inventory >= Q, np.inf, bid_depth)
    ask_depth = np.where(inventory <= -Q, np.inf, ask_depth)

    min_valid = -3.0 / k
    bid_depth = np.where(np.isfinite(bid_depth) & (bid_depth >= min_valid), bid_depth, np.inf)
    ask_depth = np.where(np.isfinite(ask_depth) & (ask_depth >= min_valid), ask_depth, np.inf)

    return bid_depth, ask_depth


def _ternary_search_max(func, lo, hi, iters=40):
    """
    Maximize a unimodal function func(x) in [lo, hi] using golden-section search.
    """
    gr = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618

    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)

    fc = func(c)
    fd = func(d)

    for _ in range(iters):
        if fc < fd:
            lo = c
            c = d
            fc = fd
            d = lo + gr * (hi - lo)
            fd = func(d)
        else:
            hi = d
            d = c
            fd = fc
            c = hi - gr * (hi - lo)
            fc = func(c)

    return (lo + hi) / 2.0


def _polish_policy_ternary(bid_table, ask_table, V, params, max_iters_per_state=5):
    """
    High-precision polish using alternating golden-section search for bid and ask depths.
    Refines the grid DP solution to near-optimal continuous depths.
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
    q_range = np.arange(-Q, Q + 1, dtype=np.float64)

    inv_cost_factor = (gamma / 2.0) * sigma * sigma * dt
    cost_arr = inv_cost_factor * q_range * q_range

    min_depth = 1e-4
    max_depth = 15.0 / k

    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 2:
            break

        Vn1 = V[n + 1]

        for qi in range(n_q):
            q = q_range[qi]
            cost = cost_arr[qi]

            d_bid = bid_table[n, qi]
            d_ask = ask_table[n, qi]

            if qi < n_q - 1:
                V_plus = Vn1[qi + 1]
            else:
                V_plus = Vn1[qi]
            if qi > 0:
                V_minus = Vn1[qi - 1]
            else:
                V_minus = Vn1[qi]

            Vn1_qi = Vn1[qi]
            D1 = V_plus - Vn1_qi
            Dm1 = V_minus - Vn1_qi
            D2 = V_plus + V_minus - 2.0 * Vn1_qi

            is_boundary_bid = (qi == n_q - 1)
            is_boundary_ask = (qi == 0)

            if is_boundary_bid and is_boundary_ask:
                continue

            if is_boundary_bid:
                # Only ask can fill
                if d_ask == np.inf:
                    continue
                lo_a = max(min_depth, d_ask * 0.3)
                hi_a = min(max_depth, d_ask * 3.0)

                def obj_ask(d_a):
                    exp_a = np.exp(-k * d_a)
                    p_a = 1.0 - np.exp(-A * exp_a * dt)
                    return d_a * p_a + Vn1_qi + p_a * Dm1 - cost

                d_ask_opt = _ternary_search_max(obj_ask, lo_a, hi_a, iters=35)
                ask_table[n, qi] = d_ask_opt

                exp_a = np.exp(-k * d_ask_opt)
                p_a = 1.0 - np.exp(-A * exp_a * dt)
                V[n, qi] = d_ask_opt * p_a + Vn1_qi + p_a * Dm1 - cost
                bid_table[n, qi] = np.inf

            elif is_boundary_ask:
                # Only bid can fill
                if d_bid == np.inf:
                    continue
                lo_b = max(min_depth, d_bid * 0.3)
                hi_b = min(max_depth, d_bid * 3.0)

                def obj_bid(d_b):
                    exp_b = np.exp(-k * d_b)
                    p_b = 1.0 - np.exp(-A * exp_b * dt)
                    return d_b * p_b + Vn1_qi + p_b * D1 - cost

                d_bid_opt = _ternary_search_max(obj_bid, lo_b, hi_b, iters=35)
                bid_table[n, qi] = d_bid_opt

                exp_b = np.exp(-k * d_bid_opt)
                p_b = 1.0 - np.exp(-A * exp_b * dt)
                V[n, qi] = d_bid_opt * p_b + Vn1_qi + p_b * D1 - cost
                ask_table[n, qi] = np.inf

            else:
                # Both can fill
                if d_bid == np.inf or d_ask == np.inf:
                    d_bid = 1.0 / k
                    d_ask = 1.0 / k

                cur_b = d_bid
                cur_a = d_ask

                lo_b = max(min_depth, d_bid * 0.3)
                hi_b = min(max_depth, d_bid * 3.0)
                lo_a = max(min_depth, d_ask * 0.3)
                hi_a = min(max_depth, d_ask * 3.0)

                for _iter in range(max_iters_per_state):
                    # Optimize bid given ask
                    def obj_bid_given_ask(d_b):
                        exp_b = np.exp(-k * d_b)
                        p_b = 1.0 - np.exp(-A * exp_b * dt)
                        exp_a = np.exp(-k * cur_a)
                        p_a = 1.0 - np.exp(-A * exp_a * dt)
                        return (d_b * p_b + Vn1_qi + p_b * D1 + p_a * Dm1 - p_b * p_a * D2 - cost)

                    new_b = _ternary_search_max(obj_bid_given_ask, lo_b, hi_b, iters=25)

                    # Optimize ask given new bid
                    def obj_ask_given_bid(d_a):
                        exp_a = np.exp(-k * d_a)
                        p_a = 1.0 - np.exp(-A * exp_a * dt)
                        exp_b = np.exp(-k * new_b)
                        p_b = 1.0 - np.exp(-A * exp_b * dt)
                        return (new_b * p_b + d_a * p_a + Vn1_qi + p_b * D1 + p_a * Dm1 - p_b * p_a * D2 - cost)

                    new_a = _ternary_search_max(obj_ask_given_bid, lo_a, hi_a, iters=25)

                    if abs(new_b - cur_b) < 1e-6 and abs(new_a - cur_a) < 1e-6:
                        cur_b = new_b
                        cur_a = new_a
                        break

                    cur_b = new_b
                    cur_a = new_a

                bid_table[n, qi] = cur_b
                ask_table[n, qi] = cur_a

                exp_b = np.exp(-k * cur_b)
                p_b = 1.0 - np.exp(-A * exp_b * dt)
                exp_a = np.exp(-k * cur_a)
                p_a = 1.0 - np.exp(-A * exp_a * dt)
                V[n, qi] = (cur_b * p_b + cur_a * p_a + Vn1_qi + p_b * D1 + p_a * Dm1 - p_b * p_a * D2 - cost)

        # Re-apply boundary conditions
        bid_table[n, n_q - 1] = np.inf
        ask_table[n, 0] = np.inf

    return bid_table, ask_table, V


def _compute_vectorized_dp(params, n_depths=300):
    """
    Vectorized DP: for each time step, compute the Bellman backup for all
    inventory states simultaneously using broadcasting.
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
    q_range = np.arange(-Q, Q + 1, dtype=np.float64)

    depth_min = 0.001
    depth_max = 12.0 / k
    depth_grid = np.geomspace(depth_min, depth_max, n_depths)

    exp_term = np.exp(-k * depth_grid)
    p_fill = 1.0 - np.exp(-A * exp_term * dt)
    depth_p = depth_grid * p_fill

    V = np.zeros((steps + 1, n_q))
    bid_table = np.zeros((steps, n_q))
    ask_table = np.zeros((steps, n_q))

    inv_cost_factor = (gamma / 2.0) * sigma * sigma * dt
    cost_arr = inv_cost_factor * q_range * q_range

    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 30:
            # Fallback: fill remaining with HMM
            for n2 in range(n, -1, -1):
                tau2 = T - n2 * dt
                base = 1.0 / k
                skew = (gamma * sigma * sigma * q_range * tau2) / (A * k)
                bid_table[n2, :] = np.clip(base + skew, depth_min, depth_max)
                ask_table[n2, :] = np.clip(base - skew, depth_min, depth_max)
                bid_table[n2, n_q - 1] = np.inf
                ask_table[n2, 0] = np.inf
            break

        Vn1 = V[n + 1]

        V_plus = np.empty(n_q)
        V_minus = np.empty(n_q)
        V_plus[:-1] = Vn1[1:]
        V_plus[-1] = Vn1[-1]
        V_minus[1:] = Vn1[:-1]
        V_minus[0] = Vn1[0]

        D1 = V_plus - Vn1
        Dm1 = V_minus - Vn1
        D2 = V_plus + V_minus - 2.0 * Vn1

        term1 = depth_p[:, None] + depth_p[None, :]

        term2 = (Vn1[None, None, :]
                 + p_fill[:, None, None] * D1[None, None, :]
                 + p_fill[None, :, None] * Dm1[None, None, :])

        pf_outer = p_fill[:, None] * p_fill[None, :]
        term3 = -(pf_outer[:, :, None] * D2[None, None, :])

        J = term1[:, :, None] + term2 + term3 - cost_arr[None, None, :]

        flat_J = J.reshape(n_depths * n_depths, n_q)
        idx_max = np.argmax(flat_J, axis=0)

        i_max = idx_max // n_depths
        j_max = idx_max % n_depths

        V[n, :] = flat_J[idx_max, np.arange(n_q)]
        bid_table[n, :] = depth_grid[i_max]
        ask_table[n, :] = depth_grid[j_max]

    # Boundary conditions
    bid_table[:, n_q - 1] = np.inf
    ask_table[:, 0] = np.inf

    # Polish pass: refine depths using ternary search
    remaining = TIME_LIMIT - 10 - (time.time() - _START_TIME)
    if remaining > 10:
        try:
            max_iters = 3
            if remaining > 60:
                max_iters = 5
            bid_table, ask_table, V = _polish_policy_ternary(bid_table, ask_table, V, params, max_iters_per_state=max_iters)
        except Exception:
            pass

    return bid_table, ask_table


_policy_cache = {}


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT:
        return _hmm_depths(params, t, inventory)

    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])

    if key not in _policy_cache:
        try:
            bid_table, ask_table = _compute_vectorized_dp(params, n_depths=300)
        except Exception:
            bid_table, ask_table = None, None
            if bid_table is None:
                return _hmm_depths(params, t, inventory)
        _policy_cache[key] = (bid_table, ask_table)
    else:
        bid_table, ask_table = _policy_cache[key]

    if bid_table is None:
        return _hmm_depths(params, t, inventory)

    steps = params['steps']
    Q = params['Q']
    dt = params['T'] / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))

    q_index = inventory + Q
    q_index = np.clip(q_index, 0, 2 * Q)

    bid_depths = bid_table[n, q_index].copy()
    ask_depths = ask_table[n, q_index].copy()

    min_valid = -3.0 / params['k']
    bid_depths = np.where((bid_depths > min_valid) & np.isfinite(bid_depths), bid_depths, np.inf)
    ask_depths = np.where((ask_depths > min_valid) & np.isfinite(ask_depths), ask_depths, np.inf)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
