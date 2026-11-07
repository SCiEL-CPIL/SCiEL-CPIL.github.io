"""
Optimal market making via pre-computed DP policy lookup.
Combines the structural DP approach with a high-resolution geometric depth grid
and vectorized matrix operations for efficiency.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 320.0  # Safety margin before 330s kill


def _heuristic_depths(params, t, inventory):
    """
    Analytical fallback: time-varying base depth with inventory skew.
    Used if DP computation times out or fails.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    Q = params['Q']
    q = inventory.astype(np.float64)
    t_remaining = np.maximum(T - t, 0.0)
    
    # Base depth is roughly 1/k
    d0 = 1.0 / k
    
    # Skew based on inventory and remaining time
    # Higher inventory -> widen bid, narrow ask to reduce inventory
    skew_scale = gamma * sigma * sigma * t_remaining
    alpha = skew_scale / (A * k) if A > 0 and k > 0 else 0.0
    alpha = min(alpha, 6.0 / k)
    
    bid_depth = d0 + alpha * q
    ask_depth = d0 - alpha * q
    
    min_depth = 0.01 / k
    max_depth = 10.0 / k
    
    bid_depth = np.clip(bid_depth, min_depth, max_depth)
    ask_depth = np.clip(ask_depth, min_depth, max_depth)
    
    # Boundaries
    bid_depth = np.where(inventory >= Q, np.inf, bid_depth)
    ask_depth = np.where(inventory <= -Q, np.inf, ask_depth)
    
    # Validity checks
    min_valid = -3.0 / k
    bid_depth = np.where(np.isfinite(bid_depth) & (bid_depth >= min_valid), bid_depth, np.inf)
    ask_depth = np.where(np.isfinite(ask_depth) & (ask_depth >= min_valid), ask_depth, np.inf)
    
    return bid_depth, ask_depth


def _compute_optimal_policy(params):
    """
    Compute optimal bid/ask depths for all (step, inventory) states via backward DP.
    Uses a geometric grid for depths and vectorized operations.
    Includes a polish pass for high-precision depth resolution.
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
    
    # Geometric grid for depths (coarse pass)
    depth_min = 0.001
    depth_max = 10.0 / k
    n_depths = 500
    depth_grid = np.geomspace(depth_min, depth_max, n_depths)
    
    # Precompute fill probabilities and expected immediate rewards
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    # r(d) = d * p(d)
    exp_term = np.exp(-k * depth_grid)
    p_fill = 1.0 - np.exp(-A * exp_term * dt)
    depth_p = depth_grid * p_fill
    
    # Value function and policy tables
    V = np.zeros((steps + 1, n_q))
    bid_table = np.zeros((steps, n_q))
    ask_table = np.zeros((steps, n_q))
    
    # Precomputed arrays for broadcasting
    P_grid = p_fill
    DP_grid = depth_p
    P_zero = np.zeros(n_depths)
    DP_zero = np.zeros(n_depths)
    
    for n in range(steps - 1, -1, -1):
        # Time check
        if time.time() - _START_TIME > TIME_LIMIT - 5:
            # Fallback: fill remaining with heuristic
            for n2 in range(n, -1, -1):
                for qi in range(n_q):
                    q = q_range[qi]
                    d = 1.0 / k + gamma * sigma**2 * q * (T - n2 * dt) / (A * k)
                    d = max(d, depth_min)
                    bid_table[n2, qi] = d
                    ask_table[n2, qi] = d
            break
        
        inv_cost_factor = (gamma / 2.0) * sigma**2 * dt
        Vn1 = V[n + 1]
        
        # Compute value differences for all q
        # V_plus[q] = Vn1[q+1] (with boundary V_plus[Q] = Vn1[Q])
        # V_minus[q] = Vn1[q-1] (with boundary V_minus[-Q] = Vn1[-Q])
        
        V_plus = np.empty(n_q)
        V_minus = np.empty(n_q)
        
        V_plus[:-1] = Vn1[1:]
        V_plus[-1] = Vn1[-1]
        
        V_minus[1:] = Vn1[:-1]
        V_minus[0] = Vn1[0]
        
        D1 = V_plus - Vn1      # V(q+1) - V(q)
        Dm1 = V_minus - Vn1    # V(q-1) - V(q)
        D2 = V_plus + V_minus - 2.0 * Vn1  # V(q+1) + V(q-1) - 2V(q)
        
        for qi in range(n_q):
            q_val = q_range[qi]
            cost = inv_cost_factor * q_val * q_val
            
            # Boundary handling
            if qi == n_q - 1:  # q = Q, no bid fills
                Pb = P_zero
                DPb = DP_zero
            else:
                Pb = P_grid
                DPb = DP_grid

            if qi == 0:  # q = -Q, no ask fills
                Pa = P_zero
                DPa = DP_zero
            else:
                Pa = P_grid
                DPa = DP_grid
            
            # Vectorized matrix construction for joint optimization
            # J[i, j] = DPb[i] + DPa[j] + Vn1[qi] + Pb[i]*D1[qi] + Pa[j]*Dm1[qi] - Pb[i]*Pa[j]*D2[qi] - cost
            
            # Shapes:
            # DPb[:, None] : (D, 1)
            # DPa[None, :] : (1, D)
            # Pb[:, None] * D1[qi] : (D, 1)
            # Pa[None, :] * Dm1[qi] : (1, D)
            # Pb[:, None] * Pa[None, :] : (D, D)
            
            term1 = DPb[:, None] + DPa[None, :]
            term2 = Vn1[qi] + Pb[:, None] * D1[qi] + Pa[None, :] * Dm1[qi]
            term3 = - (Pb[:, None] * Pa[None, :]) * D2[qi]
            
            J = term1 + term2 + term3 - cost
            
            idx_max = np.unravel_index(np.argmax(J), J.shape)
            i_max, j_max = idx_max
            
            V[n, qi] = J[i_max, j_max]
            bid_table[n, qi] = depth_grid[i_max]
            ask_table[n, qi] = depth_grid[j_max]
    
    # Polish pass: refine depths around the initially-found optima
    # For each (n, qi), create a fine grid around the coarse optimum
    polish_ratio = 1.1
    n_polish = 120
    
    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 3:
            break
        
        inv_cost_factor = (gamma / 2.0) * sigma**2 * dt
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
            q_val = q_range[qi]
            cost = inv_cost_factor * q_val * q_val
            
            # Get coarse optima
            bid_coarse = bid_table[n, qi]
            ask_coarse = ask_table[n, qi]
            
            # Create fine grids around coarse optima
            if np.isinf(bid_coarse):
                bid_fine = np.array([np.inf])
                pb_fine = np.array([0.0])
                dpb_fine = np.array([0.0])
            else:
                lo_b = max(depth_min, bid_coarse / polish_ratio)
                hi_b = min(depth_max, bid_coarse * polish_ratio)
                bid_fine = np.geomspace(lo_b, hi_b, n_polish)
                exp_b = np.exp(-k * bid_fine)
                pb_fine = 1.0 - np.exp(-A * exp_b * dt)
                dpb_fine = bid_fine * pb_fine
            
            if np.isinf(ask_coarse):
                ask_fine = np.array([np.inf])
                pa_fine = np.array([0.0])
                dpa_fine = np.array([0.0])
            else:
                lo_a = max(depth_min, ask_coarse / polish_ratio)
                hi_a = min(depth_max, ask_coarse * polish_ratio)
                ask_fine = np.geomspace(lo_a, hi_a, n_polish)
                exp_a = np.exp(-k * ask_fine)
                pa_fine = 1.0 - np.exp(-A * exp_a * dt)
                dpa_fine = ask_fine * pa_fine
            
            # Boundary handling
            if qi == n_q - 1:  # q = Q, no bid fills
                pb_fine = np.array([0.0])
                dpb_fine = np.array([0.0])
                bid_fine = np.array([np.inf])
            
            if qi == 0:  # q = -Q, no ask fills
                pa_fine = np.array([0.0])
                dpa_fine = np.array([0.0])
                ask_fine = np.array([np.inf])
            
            # Build joint objective on fine grid
            term1 = dpb_fine[:, None] + dpa_fine[None, :]
            term2 = Vn1[qi] + pb_fine[:, None] * D1[qi] + pa_fine[None, :] * Dm1[qi]
            term3 = - (pb_fine[:, None] * pa_fine[None, :]) * D2[qi]
            
            J_fine = term1 + term2 + term3 - cost
            
            idx_max = np.unravel_index(np.argmax(J_fine), J_fine.shape)
            i_max, j_max = idx_max
            
            V[n, qi] = J_fine[i_max, j_max]
            bid_table[n, qi] = bid_fine[i_max]
            ask_table[n, qi] = ask_fine[j_max]
    
    # Post-process boundaries to return inf for no quote
    bid_table[:, n_q - 1] = np.inf
    ask_table[:, 0] = np.inf

    return bid_table, ask_table


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    """
    if time.time() - _START_TIME > TIME_LIMIT:
        return _heuristic_depths(params, t, inventory)

    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])

    if key not in _policy_cache:
        bid_table, ask_table = _compute_optimal_policy(params)
        _policy_cache[key] = (bid_table, ask_table)
    else:
        bid_table, ask_table = _policy_cache[key]

    steps = params['steps']
    Q = params['Q']
    dt = params['T'] / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))

    q_index = inventory + Q
    q_index = np.clip(q_index, 0, 2 * Q)

    bid_depths = bid_table[n, q_index].copy()
    ask_depths = ask_table[n, q_index].copy()

    # Ensure valid values
    min_valid = -3.0 / params['k']
    bid_depths = np.where((bid_depths > min_valid) & np.isfinite(bid_depths), bid_depths, np.inf)
    ask_depths = np.where((ask_depths > min_valid) & np.isfinite(ask_depths), ask_depths, np.inf)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
