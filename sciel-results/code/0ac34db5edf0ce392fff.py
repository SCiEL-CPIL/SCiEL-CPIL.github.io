import numpy as np

_policy_cache = {}

def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    inv_range = list(range(-Q, Q + 1))
    n_inv = len(inv_range)
    
    V = np.zeros((steps + 1, n_inv))
    opt_bid = np.full((steps, n_inv), np.inf)
    opt_ask = np.full((steps, n_inv), np.inf)
    
    max_depth = 3.0 / k
    n_depths = 50
    depth_grid = np.linspace(0, max_depth, n_depths)
    
    for n in range(steps - 1, -1, -1):
        for inv_idx, q in enumerate(inv_range):
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            q_plus_valid = q < Q
            q_minus_valid = q > -Q
            
            v_next_q = V[n+1, inv_idx]
            v_next_q_plus = V[n+1, q+1+Q] if q_plus_valid else 0.0
            v_next_q_minus = V[n+1, q-1+Q] if q_minus_valid else 0.0
            
            # Vectorize over depth grid
            if q >= Q:
                p_b = np.zeros(n_depths)
            else:
                p_b = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
            
            if q <= -Q:
                p_a = np.zeros(n_depths)
            else:
                p_a = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
            
            # Create meshgrid for all combinations
            bid_d_grid, ask_d_grid = np.meshgrid(depth_grid, depth_grid, indexing='ij')
            p_b_grid = np.broadcast_to(p_b[:, np.newaxis], (n_depths, n_depths))
            p_a_grid = np.broadcast_to(p_a[np.newaxis, :], (n_depths, n_depths))
            
            p_both = p_b_grid * p_a_grid
            p_bid_only = p_b_grid * (1 - p_a_grid)
            p_ask_only = (1 - p_b_grid) * p_a_grid
            p_none = (1 - p_b_grid) * (1 - p_a_grid)
            
            earn = bid_d_grid * p_b_grid + ask_d_grid * p_a_grid
            future = p_both * v_next_q + p_bid_only * v_next_q_plus + p_ask_only * v_next_q_minus + p_none * v_next_q
            
            total = -cost + earn + future
            
            flat_idx = np.argmax(total)
            best_bid_idx = flat_idx // n_depths
            best_ask_idx = flat_idx % n_depths
            
            V[n, inv_idx] = total[best_bid_idx, best_ask_idx]
            opt_bid[n, inv_idx] = depth_grid[best_bid_idx]
            opt_ask[n, inv_idx] = depth_grid[best_ask_idx]
    
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key not in _policy_cache:
        _policy_cache[cache_key] = _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q)
    
    opt_bid, opt_ask = _policy_cache[cache_key]
    
    bid_depths = np.full(len(inventory), np.inf)
    ask_depths = np.full(len(inventory), np.inf)
    
    for i, q in enumerate(inventory):
        q = int(q)
        q = max(-Q, min(Q, q))
        inv_idx = q + Q
        bid_depths[i] = opt_bid[n, inv_idx]
        ask_depths[i] = opt_ask[n, inv_idx]
    
    return bid_depths, ask_depths
