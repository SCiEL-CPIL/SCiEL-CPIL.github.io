"""
Optimal market making with inventory risk.
Structural change: use a coarser grid DP for fast initialization, then perform
a continuous 2-D optimization per state using a gradient-free local search
that directly maximizes the exact Bellman objective. This avoids the
discretization error of the grid and the slow alternating 1-D searches.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 325.0  # Safety margin before 330s kill


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

    # Clip to valid range
    min_depth = 1e-6
    max_depth = 15.0 / k
    bid_depth = np.clip(bid_depth, min_depth, max_depth)
    ask_depth = np.clip(ask_depth, min_depth, max_depth)

    # Boundary: no bid fill at q=Q, no ask fill at q=-Q
    bid_depth = np.where(inventory >= Q, np.inf, bid_depth)
    ask_depth = np.where(inventory <= -Q, np.inf, ask_depth)

    # Ensure validity
    min_valid = -3.0 / k
    bid_depth = np.where(np.isfinite(bid_depth) & (bid_depth >= min_valid), bid_depth, np.inf)
    ask_depth = np.where(np.isfinite(ask_depth) & (ask_depth >= min_valid), ask_depth, np.inf)

    return bid_depth, ask_depth


def _bellman_obj(d_bid, d_ask, Vn1_qi, D1, Dm1, D2, cost, A, k, dt):
    """
    Compute the Bellman objective for given bid/ask depths.
    """
    if d_bid < 0 or d_ask < 0:
        return -np.inf
    
    exp_b = np.exp(-k * d_bid)
    exp_a = np.exp(-k * d_ask)
    p_b = 1.0 - np.exp(-A * exp_b * dt)
    p_a = 1.0 - np.exp(-A * exp_a * dt)
    
    return (d_bid * p_b + d_ask * p_a + Vn1_qi + p_b * D1 + p_a * Dm1 - p_b * p_a * D2 - cost)


def _optimize_2d(d_bid_init, d_ask_init, Vn1_qi, D1, Dm1, D2, cost, A, k, dt, is_boundary_bid, is_boundary_ask):
    """
    2-D local optimization using a simple gradient-free method (Nelder-Mead style).
    """
    min_depth = 1e-4
    max_depth = 15.0 / k
    
    if is_boundary_bid and is_boundary_ask:
        return d_bid_init, d_ask_init
    
    if is_boundary_bid:
        # Only ask depth matters
        d_ask = d_ask_init
        # 1-D search for ask
        lo = max(min_depth, d_ask * 0.3)
        hi = min(max_depth, d_ask * 3.0)
        best_d = d_ask
        best_val = -np.inf
        # Golden section search
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        fc = _bellman_obj(0.0, c, Vn1_qi, D1, Dm1, D2, cost, A, k, dt)
        fd = _bellman_obj(0.0, d, Vn1_qi, D1, Dm1, D2, cost, A, k, dt)
        for _ in range(40):
            if fc < fd:
                lo = c
                c = d
                fc = fd
                d = lo + gr * (hi - lo)
                fd = _bellman_obj(0.0, d, Vn1_qi, D1, Dm1, D2, cost, A, k, dt)
            else:
                hi = d
                d = c
                fd = fc
                c = hi - gr * (hi - lo)
                fc = _bellman_obj(0.0, c, Vn1_qi, D1, Dm1, D2, cost, A, k, dt)
        d_ask_opt = (lo + hi) / 2.0
        return np.inf, d_ask_opt
    
    if is_boundary_ask:
        # Only bid depth matters
        d_bid = d_bid_init
        lo = max(min_depth, d_bid * 0.3)
        hi = min(max_depth, d_bid * 3.0)
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        fc = _bellman_obj(c, 0.0, Vn1_qi, D1, Dm1, D2, cost, A, k, dt)
        fd = _bellman_obj(d, 0.0, Vn1_qi, D1, Dm1, D2, cost, A, k, dt)
        for _ in range(40):
            if fc < fd:
                lo = c
                c = d
                fc = fd
                d = lo + gr * (hi - lo)
                fd = _bellman_obj(d, 0.0, Vn1_qi, D1, Dm1, D2, cost, A, k, dt)
            else:
                hi = d
                d = c
                fd = fc
                c = hi - gr * (hi - lo)
                fc = _bellman_obj(c, 0.0, Vn1_qi, D1, Dm1, D2, cost, A, k, dt)
        d_bid_opt = (lo + hi) / 2.0
        return d_bid_opt, np.inf
    
    # Both depths: use a simple 2-D search
    # Start from current values
    d_b = d_bid_init
    d_a = d_ask_init
    
    # Perform a few iterations of coordinate descent with golden section
    for _iter in range(3):
        # Optimize bid given ask
        lo_b = max(min_depth, d_b * 0.3)
        hi_b = min(max_depth, d_b * 3.0)
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        c = hi_b - gr * (hi_b - lo_b)
        d = lo_b + gr * (hi_b - lo_b)
        fc = _bellman_obj(c, d_a, Vn1_qi, D1, Dm1, D2, cost, A, k, dt)
        fd = _bellman_obj(d, d_a, Vn1_qi, D1, Dm1, D2, cost, A, k, dt)
        for _ in range(30):
            if fc < fd:
                lo_b = c
                c = d
                fc = fd
                d = lo_b + gr * (hi_b - lo_b)
                fd = _bellman_obj(d, d_a, Vn1_qi, D1, Dm1, D2, cost, A, k, dt)
            else:
                hi_b = d
                d = c
                fd = fc
                c = hi_b - gr * (hi_b - lo_b)
                fc = _bellman_obj(c, d_a, Vn1_qi, D1, Dm1, D2, cost, A, k, dt)
        d_b = (lo_b + hi_b) / 2.0
        
        # Optimize ask given bid
        lo_a = max(min_depth, d_a * 0.3)
        hi_a = min(max_depth, d_a * 3.0)
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        c = hi_a - gr * (hi_a - lo_a)
        d = lo_a + gr * (hi_a - lo_a)
        fc = _bellman_obj(d_b, c, Vn1_qi, D1, Dm1, D2, cost, A, k, dt)
        fd = _bellman_obj(d_b, d, Vn1_qi, D1, Dm1, D2, cost, A, k, dt)
        for _ in range(30):
            if fc < fd:
                lo_a = c
                c = d
                fc = fd
                d = lo_a + gr * (hi_a - lo_a)
                fd = _bellman_obj(d_b, d, Vn1_qi, D1, Dm1, D2, cost, A, k, dt)
            else:
                hi_a = d
                d = c
                fd = fc
                c = hi_a - gr * (hi_a - lo_a)
                fc = _bellman_obj(d_b, c, Vn1_qi, D1, Dm1, D2, cost, A, k, dt)
        d_a = (lo_a + hi_a) / 2.0
    
    return d_b, d_a


def _compute_vectorized_dp(params, n_depths=150):
    """
    Vectorized DP with coarser grid, followed by continuous optimization.
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
        if time.time() - _START_TIME > TIME_LIMIT - 60:
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

        Vn1 = V[n + 1]  # shape (n_q,)

        V_plus = np.empty(n_q)
        V_minus = np.empty(n_q)
        V_plus[:-1] = Vn1[1:]
        V_plus[-1] = Vn1[-1]
        V_minus[1:] = Vn1[:-1]
        V_minus[0] = Vn1[0]

        D1 = V_plus - Vn1       # shape (n_q,)
        Dm1 = V_minus - Vn1     # shape (n_q,)
        D2 = V_plus + V_minus - 2.0 * Vn1  # shape (n_q,)

        # Build the J tensor
        term1 = depth_p[:, None] + depth_p[None, :]  # (n_depths, n_depths)

        term2 = (Vn1[None, None, :]
                 + p_fill[:, None, None] * D1[None, None, :]
                 + p_fill[None, :, None] * Dm1[None, None, :])  # (n_depths, n_depths, n_q)

        pf_outer = p_fill[:, None] * p_fill[None, :]  # (n_depths, n_depths)
        term3 = -(pf_outer[:, :, None] * D2[None, None, :])  # (n_depths, n_depths, n_q)

        J = term1[:, :, None] + term2 + term3 - cost_arr[None, None, :]  # (n_depths, n_depths, n_q)

        flat_J = J.reshape(n_depths * n_depths, n_q)
        idx_max = np.argmax(flat_J, axis=0)  # shape (n_q,)

        i_max = idx_max // n_depths
        j_max = idx_max % n_depths

        V[n, :] = flat_J[idx_max, np.arange(n_q)]
        bid_table[n, :] = depth_grid[i_max]
        ask_table[n, :] = depth_grid[j_max]

    # Boundary conditions
    bid_table[:, n_q - 1] = np.inf
    ask_table[:, 0] = np.inf

    # Continuous optimization pass
    remaining = TIME_LIMIT - 30 - (time.time() - _START_TIME)
    if remaining > 5:
        for n in range(steps - 1, -1, -1):
            if time.time() - _START_TIME > TIME_LIMIT - 30:
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

            for qi in range(n_q):
                is_boundary_bid = (qi == n_q - 1)
                is_boundary_ask = (qi == 0)
                
                if is_boundary_bid and is_boundary_ask:
                    continue
                
                d_bid_init = bid_table[n, qi]
                d_ask_init = ask_table[n, qi]
                
                if not np.isfinite(d_bid_init):
                    d_bid_init = 1.0 / k
                if not np.isfinite(d_ask_init):
                    d_ask_init = 1.0 / k
                
                d_bid_opt, d_ask_opt = _optimize_2d(
                    d_bid_init, d_ask_init, Vn1[qi], D1[qi], Dm1[qi], D2[qi],
                    cost_arr[qi], A, k, dt, is_boundary_bid, is_boundary_ask
                )
                
                bid_table[n, qi] = d_bid_opt
                ask_table[n, qi] = d_ask_opt
                
                # Update V
                if is_boundary_bid and is_boundary_ask:
                    continue
                elif is_boundary_bid:
                    V[n, qi] = _bellman_obj(np.inf, d_ask_opt, Vn1[qi], D1[qi], Dm1[qi], D2[qi], cost_arr[qi], A, k, dt)
                elif is_boundary_ask:
                    V[n, qi] = _bellman_obj(d_bid_opt, np.inf, Vn1[qi], D1[qi], Dm1[qi], D2[qi], cost_arr[qi], A, k, dt)
                else:
                    V[n, qi] = _bellman_obj(d_bid_opt, d_ask_opt, Vn1[qi], D1[qi], Dm1[qi], D2[qi], cost_arr[qi], A, k, dt)
            
            # Re-apply boundary conditions
            bid_table[n, n_q - 1] = np.inf
            ask_table[n, 0] = np.inf

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
            bid_table, ask_table = _compute_vectorized_dp(params, n_depths=150)
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
