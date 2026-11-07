"""
Optimal market making with inventory risk.
Uses a vectorized backward DP with alternating 1D ternary search for depth optimization.
This approach avoids the memory explosion of 2D grid searches and converges to the
local optimum efficiently by exploiting the unimodality of the 1D sub-problems.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 320.0  # safety margin before 330s kill


def _hmm_depths(params, t, inventory):
    """
    Hopenhaver-Sussmann closed-form approximation.
    Used as initialization and fallback.
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


def _compute_optimal_dp(params):
    """
    Computes the optimal policy using backward induction.
    Uses vectorized ternary search for the 1D depth optimizations,
    alternating between bid and ask to handle the coupling.
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

    # Precompute constants
    inv_cost_factor = (gamma / 2.0) * sigma * sigma * dt
    cost_arr = inv_cost_factor * q_range * q_range  # shape (n_q,)

    # Initialize V and policy tables
    V = np.zeros((steps + 1, n_q))
    bid_table = np.zeros((steps, n_q))
    ask_table = np.zeros((steps, n_q))

    # Depth bounds for search
    d_min = 1e-4
    d_max = 15.0 / k
    
    # Number of ternary search iterations
    n_iter = 60

    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 10:
            # Fallback: fill remaining with HMM
            for n2 in range(n, -1, -1):
                tau2 = T - n2 * dt
                base = 1.0 / k
                skew = (gamma * sigma * sigma * q_range * tau2) / (A * k)
                bid_table[n2, :] = np.clip(base + skew, d_min, d_max)
                ask_table[n2, :] = np.clip(base - skew, d_min, d_max)
                bid_table[n2, n_q - 1] = np.inf
                ask_table[n2, 0] = np.inf
            break

        Vn1 = V[n + 1]

        # Compute value differences
        V_plus = np.empty(n_q)
        V_minus = np.empty(n_q)
        V_plus[:-1] = Vn1[1:]
        V_plus[-1] = Vn1[-1]
        V_minus[1:] = Vn1[:-1]
        V_minus[0] = Vn1[0]

        D1 = V_plus - Vn1       # \Delta_+ V
        Dm1 = V_minus - Vn1     # \Delta_- V
        D2 = V_plus + V_minus - 2.0 * Vn1  # \Delta_2 V

        # Initialize depths using HMM approximation for this step
        tau = T - n * dt
        base_d = 1.0 / k
        skew = (gamma * sigma * sigma * q_range * tau) / (A * k)
        
        d_bid_curr = np.clip(base_d + skew, d_min, d_max)
        d_ask_curr = np.clip(base_d - skew, d_min, d_max)

        # Handle boundaries: q=Q (no bid), q=-Q (no ask)
        # We will mask these out in the optimization or set them directly.
        is_bid_boundary = (q_range == Q)
        is_ask_boundary = (q_range == -Q)

        # Active states for bid optimization (not at Q)
        active_bid = ~is_bid_boundary
        # Active states for ask optimization (not at -Q)
        active_ask = ~is_ask_boundary
        
        # We alternate optimizing bid and ask.
        # Since the state space is small (61), we can just run the vectorized ternary search
        # on the active subsets.
        
        # For simplicity and speed, we run the search on ALL states but mask the objective
        # to -inf for boundary states where that side is invalid, or just skip them.
        # Better: Just run on all, and fix boundaries at the end.
        # But the coupling term D2 is 0 at boundaries? No, D2 is defined for all.
        # At q=Q, bid depth is inf. So p_bid = 0.
        # The term involving p_bid vanishes.
        
        # Let's run the alternating optimization.
        # We define the objective function J(d_bid, d_ask) for each state q.
        
        # To vectorize ternary search, we maintain lo/hi bounds for each state.
        # We will do 2 passes: one for bid, one for ask, repeated a few times.
        # Or just one pass of bid then ask is often sufficient if HMM is good.
        # Let's do 3 iterations of (Bid Opt, Ask Opt) for robustness.
        
        for _ in range(3):
            # --- Optimize Bid Depths ---
            # For each q, maximize over d_b:
            # f(d_b) = p_b(d_b) * (d_b + D1[q] - p_a_curr * D2[q]) + const
            # where p_a_curr is the current ask probability.
            
            p_a_curr = 1.0 - np.exp(-A * np.exp(-k * d_ask_curr) * dt)
            
            # Term inside bracket for bid: K_b = D1 - p_a * D2
            K_b = D1 - p_a_curr * D2
            
            # Objective for bid: g(d_b) = p_b(d_b) * (d_b + K_b)
            # We want to maximize this.
            
            # Initialize search bounds around current estimate
            lo_b = np.maximum(d_min, d_bid_curr * 0.5)
            hi_b = np.minimum(d_max, d_bid_curr * 2.0 + 1.0/k)
            
            # Ternary search vectorized
            for _iter in range(n_iter):
                m1 = lo_b + (hi_b - lo_b) / 3.0
                m2 = hi_b - (hi_b - lo_b) / 3.0
                
                p_m1 = 1.0 - np.exp(-A * np.exp(-k * m1) * dt)
                p_m2 = 1.0 - np.exp(-A * np.exp(-k * m2) * dt)
                
                val_m1 = p_m1 * (m1 + K_b)
                val_m2 = p_m2 * (m2 + K_b)
                
                # Update bounds
                # If val_m1 < val_m2, max is in [m1, hi]
                # If val_m1 > val_m2, max is in [lo, m2]
                lo_b = np.where(val_m1 < val_m2, m1, lo_b)
                hi_b = np.where(val_m1 > val_m2, m2, hi_b)
                
                # Handle boundaries where bid is not allowed (q=Q)
                # We can just set their depth to 0 or ignore, we fix at end.
                
            d_bid_new = (lo_b + hi_b) / 2.0
            d_bid_new = np.clip(d_bid_new, d_min, d_max)
            
            # --- Optimize Ask Depths ---
            p_b_new = 1.0 - np.exp(-A * np.exp(-k * d_bid_new) * dt)
            
            # Term inside bracket for ask: K_a = Dm1 - p_b * D2
            K_a = Dm1 - p_b_new * D2
            
            # Objective for ask: g(d_a) = p_a(d_a) * (d_a + K_a)
            
            lo_a = np.maximum(d_min, d_ask_curr * 0.5)
            hi_a = np.minimum(d_max, d_ask_curr * 2.0 + 1.0/k)
            
            for _iter in range(n_iter):
                m1 = lo_a + (hi_a - lo_a) / 3.0
                m2 = hi_a - (hi_a - lo_a) / 3.0
                
                p_m1 = 1.0 - np.exp(-A * np.exp(-k * m1) * dt)
                p_m2 = 1.0 - np.exp(-A * np.exp(-k * m2) * dt)
                
                val_m1 = p_m1 * (m1 + K_a)
                val_m2 = p_m2 * (m2 + K_a)
                
                lo_a = np.where(val_m1 < val_m2, m1, lo_a)
                hi_a = np.where(val_m1 > val_m2, m2, hi_a)
                
            d_ask_new = (lo_a + hi_a) / 2.0
            d_ask_new = np.clip(d_ask_new, d_min, d_max)
            
            d_bid_curr = d_bid_new
            d_ask_curr = d_ask_new

        # Store results
        bid_table[n, :] = d_bid_curr
        ask_table[n, :] = d_ask_curr
        
        # Calculate optimal V[n, q]
        p_b = 1.0 - np.exp(-A * np.exp(-k * d_bid_curr) * dt)
        p_a = 1.0 - np.exp(-A * np.exp(-k * d_ask_curr) * dt)
        
        # V_n(q) = d_b p_b + d_a p_a + V_{n+1}(q) + p_b D1 + p_a Dm1 - p_b p_a D2 - cost
        V[n, :] = (d_bid_curr * p_b + d_ask_curr * p_a + Vn1 + 
                   p_b * D1 + p_a * Dm1 - p_b * p_a * D2 - cost_arr)

    # Apply boundary conditions: no bid at Q, no ask at -Q
    bid_table[:, n_q - 1] = np.inf
    ask_table[:, 0] = np.inf
    
    # Recalculate V for boundaries? 
    # At q=Q, p_b=0. V_n(Q) = d_a p_a + V_{n+1}(Q) + p_a Dm1 - cost.
    # Our formula above used p_b from the table, which is non-zero for q=Q before masking.
    # We should correct V for boundaries.
    
    # Correct V for q=Q (index n_q-1)
    # p_b is effectively 0.
    p_a_Q = 1.0 - np.exp(-A * np.exp(-k * ask_table[:, n_q-1]) * dt)
    # D1 at Q is V(Q+1)-V(Q) but Q+1 doesn't exist, so in our DP V_plus[Q] = V[Q].
    # So D1[Q] = 0.
    # Dm1[Q] = V(Q-1) - V(Q).
    # D2[Q] = V(Q+1) + V(Q-1) - 2V(Q) = V(Q) + V(Q-1) - 2V(Q) = V(Q-1) - V(Q) = Dm1[Q].
    # Wait, in the loop:
    # V_plus[-1] = Vn1[-1] -> D1[-1] = 0.
    # V_minus[-1] = Vn1[-2] -> Dm1[-1] = Vn1[-2] - Vn1[-1].
    # D2[-1] = 0 + Dm1[-1] - 0 = Dm1[-1].
    
    # V_n(Q) = d_a p_a + V_{n+1}(Q) + 0 + p_a Dm1[Q] - 0 - cost[Q]
    # Note: p_b * D1 = 0, p_b p_a D2 = 0.
    
    for n in range(steps):
        d_a = ask_table[n, n_q-1]
        if np.isinf(d_a):
            p_a = 0.0
        else:
            p_a = 1.0 - np.exp(-A * np.exp(-k * d_a) * dt)
            
        Vn1_Q = V[n+1, n_q-1]
        Dm1_Q = V[n+1, n_q-2] - V[n+1, n_q-1]
        cost_Q = cost_arr[n_q-1]
        
        V[n, n_q-1] = d_a * p_a + Vn1_Q + p_a * Dm1_Q - cost_Q

    # Correct V for q=-Q (index 0)
    for n in range(steps):
        d_b = bid_table[n, 0]
        if np.isinf(d_b):
            p_b = 0.0
        else:
            p_b = 1.0 - np.exp(-A * np.exp(-k * d_b) * dt)
            
        Vn1_mQ = V[n+1, 0]
        D1_mQ = V[n+1, 1] - V[n+1, 0]
        cost_mQ = cost_arr[0]
        
        V[n, 0] = d_b * p_b + Vn1_mQ + p_b * D1_mQ - cost_mQ

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
            bid_table, ask_table = _compute_optimal_dp(params)
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
