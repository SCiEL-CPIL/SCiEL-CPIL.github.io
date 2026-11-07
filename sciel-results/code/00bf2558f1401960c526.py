"""Optimal market making with inventory risk - DP Solver."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0
_POLICY_CACHE = {}

def _get_policy(sigma, A, k, gamma, T, steps, Q):
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
        
    dt = T / steps
    n_steps = steps
    n_inv = 2 * Q + 1
    inv_range = np.arange(-Q, Q + 1)
    
    # V[t, q_idx]
    V = np.zeros((n_steps + 1, n_inv))
    
    # Optimal depths: bid_depth[t, q_idx], ask_depth[t, q_idx]
    # If no quote, depth is np.inf
    bid_depths = np.full((n_steps, n_inv), np.inf)
    ask_depths = np.full((n_steps, n_inv), np.inf)
    
    # Grid for depth search
    # Max depth: if depth is too large, prob -> 0, value -> 0.
    # Min depth: 0.
    # Let's search in [0, 5.0/k]
    d_max = 5.0 / k
    num_d = 200
    d_grid = np.linspace(0.0, d_max, num_d)
    
    # Precompute exp(-k * d_grid)
    exp_kd_grid = np.exp(-k * d_grid)
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    p_grid = 1.0 - np.exp(-A * exp_kd_grid * dt)
    
    for t in range(n_steps - 1, -1, -1):
        # Cost at this step
        q_vals = inv_range.astype(np.float64)
        cost = (gamma / 2.0) * sigma * sigma * q_vals * q_vals * dt
        
        for q_idx in range(n_inv):
            q = inv_range[q_idx]
            
            # Boundary checks
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            best_bid_val = 0.0
            best_bid_d = np.inf
            best_ask_val = 0.0
            best_ask_d = np.inf
            
            if can_bid:
                delta_v_b = V[t+1, q_idx+1] - V[t+1, q_idx]
                # Maximize p(d) * (d + delta_v_b)
                vals = p_grid * (d_grid + delta_v_b)
                max_idx = np.argmax(vals)
                best_bid_val = vals[max_idx]
                best_bid_d = d_grid[max_idx]
            else:
                best_bid_val = 0.0
                best_bid_d = np.inf
                
            if can_ask:
                delta_v_a = V[t+1, q_idx-1] - V[t+1, q_idx]
                # Maximize p(d) * (d + delta_v_a)
                vals = p_grid * (d_grid + delta_v_a)
                max_idx = np.argmax(vals)
                best_ask_val = vals[max_idx]
                best_ask_d = d_grid[max_idx]
            else:
                best_ask_val = 0.0
                best_ask_d = np.inf
            
            V[t, q_idx] = best_bid_val + best_ask_val - cost
            bid_depths[t, q_idx] = best_bid_d
            ask_depths[t, q_idx] = best_ask_d
            
    _POLICY_CACHE[key] = (bid_depths, ask_depths, inv_range)
    return _POLICY_CACHE[key]

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Determine step index
    # t = n * dt, so n = t / dt = t * steps / T
    n = int(round(t * steps / T))
    if n >= steps:
        n = steps - 1
        
    bid_depths, ask_depths, inv_range = _get_policy(sigma, A, k, gamma, T, steps, Q)
    
    # Map inventory to index
    # inventory is in [-Q, Q]
    # index = inventory + Q
    q_idx = inventory + Q
    
    # Extract depths for this step
    # Handle potential out of bounds just in case, though inventory should be valid
    valid_idx = (q_idx >= 0) & (q_idx < len(inv_range))
    
    res_bid = np.full(len(inventory), np.inf)
    res_ask = np.full(len(inventory), np.inf)
    
    res_bid[valid_idx] = bid_depths[n, q_idx[valid_idx]]
    res_ask[valid_idx] = ask_depths[n, q_idx[valid_idx]]
    
    return res_bid, res_ask
# EVOLVE-BLOCK-END
