import numpy as np
import time

_policy_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 310.0

def _heuristic_depths(params, t, inventory):
    """Analytical fallback."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    Q = params['Q']
    q = inventory.astype(np.float64)
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
    min_valid = -3.0 / k
    bid_depth = np.where(np.isfinite(bid_depth) & (bid_depth >= min_valid), bid_depth, np.inf)
    ask_depth = np.where(np.isfinite(ask_depth) & (ask_depth >= min_valid), ask_depth, np.inf)
    return bid_depth, ask_depth

def _compute_optimal_policy(params):
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
    q_sq = q_range.astype(np.float64) ** 2

    # Grid definition
    # Optimal depths are typically small. 
    # Start with a coarse grid.
    depth_min = 1e-4 / k
    depth_max = 10.0 / k
    n_depths_coarse = 100
    depth_grid_coarse = np.geomspace(depth_min, depth_max, n_depths_coarse)
    
    # Precompute for coarse grid
    exp_c = np.exp(-k * depth_grid_coarse)
    p_c = 1.0 - np.exp(-A * exp_c * dt)
    dp_c = depth_grid_coarse * p_c
    
    V = np.zeros((steps + 1, n_q))
    bid_table = np.zeros((steps, n_q))
    ask_table = np.zeros((steps, n_q))
    
    # DP Pass 1: Coarse
    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 30:
            break
            
        Vn1 = V[n + 1]
        
        # Prepare next state values for bid and ask fills
        # V_bid_next[q] = V(q+1) if q < Q else V(q) (no change if no fill possible, but we mask)
        # Actually, if q=Q, bid fill is impossible. The term p_b * (V(q+1)-V(q)) should be 0.
        # We can just compute D1 and Dm for all q, and mask the J matrix.
        
        # V_plus: V(q+1) for q < Q, V(Q) for q=Q (dummy)
        V_plus = np.empty(n_q)
        V_plus[:-1] = Vn1[1:]
        V_plus[-1] = Vn1[-1]
        
        # V_minus: V(q-1) for q > -Q, V(-Q) for q=-Q (dummy)
        V_minus = np.empty(n_q)
        V_minus[1:] = Vn1[:-1]
        V_minus[0] = Vn1[0]
        
        D1 = V_plus - Vn1
        Dm1 = V_minus - Vn1
        D2 = V_plus + V_minus - 2.0 * Vn1
        
        cost = (gamma / 2.0) * sigma**2 * dt * q_sq
        
        # Vectorized J computation
        # p_c shape (N_d,), dp_c shape (N_d,)
        # D1, Dm1, D2, cost, Vn1 shape (N_q,)
        
        # Term 1: dp_b[i] + dp_a[j]
        # Shape (N_d, N_d, 1)
        T1 = dp_c[:, None] + dp_c[None, :]
        
        # Term 2: Vn1[q] + p_b[i] D1[q] + p_a[j] Dm1[q]
        # p_b[i] D1[q] -> (N_d, 1, N_q)
        # p_a[j] Dm1[q] -> (1, N_d, N_q)
        # Vn1[q] -> (1, 1, N_q)
        T2 = Vn1[None, None, :] + p_c[:, None, None] * D1[None, None, :] + p_c[None, :, None] * Dm1[None, None, :]
        
        # Term 3: - p_b[i] p_a[j] D2[q]
        # p_b[i] p_a[j] -> (N_d, N_d, 1)
        # D2[q] -> (1, 1, N_q)
        T3 = - (p_c[:, None, None] * p_c[None, :, None]) * D2[None, None, :]
        
        J = T1[:, :, None] + T2 + T3 - cost[None, None, :]
        
        # Apply boundary masks
        # q = Q (index n_q-1): No bid. Set all J[i, j, Q] to -inf for i > 0? 
        # Actually, we want to force i to be "inf". 
        # If we just set J[:, :, n_q-1] to -inf except for i=0 (depth=0, p=0), 
        # the argmax will pick i=0. 
        # But depth=0 is not inf. 
        # We will post-process the table to set inf for boundaries.
        # So in DP, we just need to ensure the *value* is correct.
        # If we allow i=0 (depth ~ 0), p~0, the value is correct.
        # So we just need to mask out "large" depths? No, any depth is valid in the DP math, 
        # as long as we interpret the result correctly.
        # However, if we pick a large depth, it's a valid quote. 
        # The constraint "no bid fill at q=Q" is a model constraint. 
        # If the model enforces it, the probability is 0. 
        # My DP assumes probability is p(d). 
        # So for q=Q, I MUST force p_b = 0.
        # So I set J[i, j, n_q-1] = -inf for all i such that p_c[i] > 0.
        # Since p_c[0] is very small but not 0, I should probably just set the whole column to -inf 
        # and handle the "no quote" case separately? 
        # No, "no quote" means depth = inf. 
        # In the grid, we don't have inf. 
        # We can add a special "inf" bucket? 
        # Or simply, for q=Q, we only optimize over ask depth j. 
        # We can do this by setting J[:, :, n_q-1] = -inf, then 
        # J[0, :, n_q-1] = J[0, :, n_q-1] (allowing i=0, which is depth~0, p~0).
        # This is an approximation. 
        # Better: Add a specific "no quote" option to the grid? 
        # That complicates the vectorization.
        
        # Let's just mask out i>0 for q=Q and j>0 for q=-Q.
        # i=0 corresponds to the smallest depth. p is small.
        # It's a good approximation for "no quote" if the smallest depth is small enough.
        
        J[:, :, n_q-1] = -np.inf
        J[0, :, n_q-1] = J[0, :, n_q-1] # Allow i=0
        
        J[:, :, 0] = -np.inf
        J[:, 0, 0] = J[:, 0, 0] # Allow j=0
        
        # Argmax
        # Flatten the first two dimensions
        J_flat = J.reshape(n_depths_coarse * n_depths_coarse, n_q)
        idx_flat = np.argmax(J_flat, axis=0)
        
        i_max = idx_flat // n_depths_coarse
        j_max = idx_flat % n_depths_coarse
        
        # Get values
        V[n] = J_flat[idx_flat, np.arange(n_q)]
        
        bid_table[n] = depth_grid_coarse[i_max]
        ask_table[n] = depth_grid_coarse[j_max]
        
    # Check time
    if time.time() - _START_TIME > TIME_LIMIT - 20:
        # Fallback or just return coarse
        bid_table[:, n_q-1] = np.inf
        ask_table[:, 0] = np.inf
        return bid_table, ask_table

    # Refinement: Find range of optimal depths
    # Exclude boundaries
    valid_bid = bid_table[:, 1:n_q-1]
    valid_ask = ask_table[:, 1:n_q-1]
    
    # Percentiles to avoid outliers
    bid_min = np.percentile(valid_bid, 1)
    bid_max = np.percentile(valid_bid, 99)
    ask_min = np.percentile(valid_ask, 1)
    ask_max = np.percentile(valid_ask, 99)
    
    # Expand range
    margin = 0.2
    new_min = max(1e-4/k, min(bid_min, ask_min) * (1-margin))
    new_max = min(10.0/k, max(bid_max, ask_max) * (1+margin))
    
    # Ensure new range is valid
    if new_min >= new_max:
        new_min = 1e-4/k
        new_max = 10.0/k
        
    n_depths_fine = 400
    depth_grid_fine = np.geomspace(new_min, new_max, n_depths_fine)
    
    # Precompute for fine grid
    exp_f = np.exp(-k * depth_grid_fine)
    p_f = 1.0 - np.exp(-A * exp_f * dt)
    dp_f = depth_grid_fine * p_f
    
    # DP Pass 2: Fine
    V = np.zeros((steps + 1, n_q))
    bid_table = np.zeros((steps, n_q))
    ask_table = np.zeros((steps, n_q))
    
    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 10:
            break
            
        Vn1 = V[n + 1]
        
        V_plus = np.empty(n_q)
        V_plus[:-1] = Vn1[1:]
        V_plus[-1] = Vn1[-1]
        
        V_minus = np.empty(n_q)
        V_minus[1:] = Vn1[:-1]
        V_minus[0] = Vn1[0]
        
        D1 = V_plus - Vn1
        Dm1 = V_minus - Vn1
        D2 = V_plus + V_minus - 2.0 * Vn1
        
        cost = (gamma / 2.0) * sigma**2 * dt * q_sq
        
        T1 = dp_f[:, None] + dp_f[None, :]
        T2 = Vn1[None, None, :] + p_f[:, None, None] * D1[None, None, :] + p_f[None, :, None] * Dm1[None, None, :]
        T3 = - (p_f[:, None, None] * p_f[None, :, None]) * D2[None, None, :]
        
        J = T1[:, :, None] + T2 + T3 - cost[None, None, :]
        
        # Masks
        J[:, :, n_q-1] = -np.inf
        J[0, :, n_q-1] = J[0, :, n_q-1]
        
        J[:, :, 0] = -np.inf
        J[:, 0, 0] = J[:, 0, 0]
        
        J_flat = J.reshape(n_depths_fine * n_depths_fine, n_q)
        idx_flat = np.argmax(J_flat, axis=0)
        
        i_max = idx_flat // n_depths_fine
        j_max = idx_flat % n_depths_fine
        
        V[n] = J_flat[idx_flat, np.arange(n_q)]
        
        bid_table[n] = depth_grid_fine[i_max]
        ask_table[n] = depth_grid_fine[j_max]
        
    # Post-process boundaries
    bid_table[:, n_q-1] = np.inf
    ask_table[:, 0] = np.inf
    
    return bid_table, ask_table

def quote(t, inventory, mid, params):
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

    min_valid = -3.0 / params['k']
    bid_depths = np.where((bid_depths > min_valid) & np.isfinite(bid_depths), bid_depths, np.inf)
    ask_depths = np.where((ask_depths > min_valid) & np.isfinite(ask_depths), ask_depths, np.inf)

    return bid_depths, ask_depths
