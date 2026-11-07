import numpy as np
import time

# Global cache for precomputed solutions
_solution_cache = {}

def _precompute_solution(sigma, A, k, gamma, T, steps, Q):
    start_time = time.time()
    deadline = 300.0  # 300s limit, keep margin
    
    dt = T / steps
    
    # Precompute optimal p for a grid of C (linear coefficient)
    # C represents the marginal value difference V(q+1) - V(q)
    # Estimate range of C
    # Max holding cost per step ~ 0.5 * gamma * sigma^2 * Q^2 * dt
    # Total max cost ~ 0.5 * gamma * sigma^2 * Q^2
    max_cost = 0.5 * gamma * sigma**2 * Q**2
    # C can be positive or negative.
    # If V(q+1) > V(q), C > 0. We want to increase q? No, if V(q+1) is higher, 
    # moving to q+1 is good. But bid fill increases q.
    # So if C > 0, we prefer bid fill?
    # Objective: p(d + C). If C is large positive, we want p=1?
    # If C is large negative, we want p=0?
    # Let's set a safe range.
    C_min, C_max = -2 * max_cost, 2 * max_cost
    
    N_C = 2000
    C_grid = np.linspace(C_min, C_max, N_C)
    
    # p_grid for optimization
    # p_max = 1 - exp(-A * exp(-k * d_min) * dt)
    # d_min = 0 => p_max = 1 - exp(-A * dt)
    p_max = 1 - np.exp(-A * dt)
    
    N_P = 2000
    p_grid = np.linspace(0, p_max, N_P)
    
    # Calculate depths d(p)
    # d = -1/k * log( -log(1-p) / (A*dt) )
    # Handle p=0 and p=p_max carefully
    eps = 1e-15
    p_safe = np.clip(p_grid, eps, p_max - eps)
    
    # -log(1-p)
    log_term = -np.log(1 - p_safe)
    # ratio = log_term / (A * dt)
    ratio = log_term / (A * dt)
    ratio = np.clip(ratio, 1e-15, 1.0)
    d_grid = -np.log(ratio) / k
    
    # f(p, C) = p * (d(p) + C)
    # We want argmax_p f(p) for each C.
    # vals[i, j] = p_grid[j] * (d_grid[j] + C_grid[i])
    # Broadcasting: (N_C, 1) + (1, N_P) -> (N_C, N_P)
    vals = p_grid[None, :] * (d_grid[None, :] + C_grid[:, None])
    
    # Find argmax along axis 1
    idx_max = np.argmax(vals, axis=1)
    p_opt_grid = p_grid[idx_max]
    
    # Now solve DP
    # V[n, q] for q in -Q..Q
    # Size: steps x (2Q+1)
    V = np.zeros((steps, 2*Q + 1))
    # Optimal depths
    d_b_opt = np.full((steps, 2*Q + 1), np.inf)
    d_a_opt = np.full((steps, 2*Q + 1), np.inf)
    
    # Initialize V[steps-1] = 0
    
    for n in range(steps - 2, -1, -1):
        if time.time() > deadline - 10:
             # Fallback: use last computed or simple strategy
             # Fill remaining with simple strategy
             depth = 1.0 / k
             d_b_opt[n:, :] = depth
             d_a_opt[n:, :] = depth
             break
             
        V_next = V[n+1]
        
        # Compute deltas for all q
        # q index 0..2Q corresponds to -Q..Q
        # delta_plus[i] = V_next[i+1] - V_next[i] for i < 2Q
        # delta_minus[i] = V_next[i-1] - V_next[i] for i > 0
        
        delta_plus = np.zeros(2*Q + 1)
        delta_minus = np.zeros(2*Q + 1)
        
        if 2*Q > 0:
            delta_plus[:-1] = V_next[1:] - V_next[:-1]
            # For i = 2Q, delta_plus is not used (no bid)
            
            delta_minus[1:] = V_next[:-1] - V_next[1:]
            # For i = 0, delta_minus is not used (no ask)
        
        # Look up optimal p_b and p_a
        # Clip deltas to C_grid range
        dp_clipped = np.clip(delta_plus, C_min, C_max)
        dm_clipped = np.clip(delta_minus, C_min, C_max)
        
        # Searchsorted to find index in C_grid
        idx_b = np.searchsorted(C_grid, dp_clipped)
        idx_a = np.searchsorted(C_grid, dm_clipped)
        
        # Ensure indices are within bounds
        idx_b = np.clip(idx_b, 0, N_C - 1)
        idx_a = np.clip(idx_a, 0, N_C - 1)
        
        p_b = p_opt_grid[idx_b]
        p_a = p_opt_grid[idx_a]
        
        # Set p_b = 0 at q=Q (i=2Q)
        if 2*Q >= 0:
            p_b[2*Q] = 0.0
            p_a[0] = 0.0
            
        # Compute depths
        # d = -1/k * log( -log(1-p) / (A*dt) )
        # If p == 0, d = inf
        d_b = np.full(2*Q + 1, np.inf)
        d_a = np.full(2*Q + 1, np.inf)
        
        mask_b = p_b > 0
        mask_a = p_a > 0
        
        if np.any(mask_b):
            p_b_safe = np.clip(p_b[mask_b], eps, p_max - eps)
            log_term_b = -np.log(1 - p_b_safe)
            ratio_b = log_term_b / (A * dt)
            ratio_b = np.clip(ratio_b, 1e-15, 1.0)
            d_b_vals = -np.log(ratio_b) / k
            d_b[mask_b] = d_b_vals
            
        if np.any(mask_a):
            p_a_safe = np.clip(p_a[mask_a], eps, p_max - eps)
            log_term_a = -np.log(1 - p_a_safe)
            ratio_a = log_term_a / (A * dt)
            ratio_a = np.clip(ratio_a, 1e-15, 1.0)
            d_a_vals = -np.log(ratio_a) / k
            d_a[mask_a] = d_a_vals
            
        d_b_opt[n, :] = d_b
        d_a_opt[n, :] = d_a
        
        # Compute V_curr
        # Reward: p_b * d_b + p_a * d_a
        # Handle 0 * inf = nan
        reward_b = np.where(p_b > 0, p_b * d_b, 0.0)
        reward_a = np.where(p_a > 0, p_a * d_a, 0.0)
        
        # Cost
        q_vals = np.arange(-Q, Q + 1)
        cost = 0.5 * gamma * sigma**2 * q_vals**2 * dt
        
        # Expected Value
        # E[V_next] = V_next + p_b * delta_plus + p_a * delta_minus - p_b * p_a * (delta_plus + delta_minus)
        # Note: delta_plus[2Q] and delta_minus[0] are 0, and p_b[2Q]=0, p_a[0]=0, so terms vanish correctly.
        # But we need to be careful with the -p_b p_a term.
        # If p_b=0 or p_a=0, the term is 0.
        
        ev_term = V_next + p_b * delta_plus + p_a * delta_minus - p_b * p_a * (delta_plus + delta_minus)
        
        V_curr = reward_b + reward_a - cost + ev_term
        V[n] = V_curr
        
    return d_b_opt, d_a_opt

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _solution_cache:
        try:
            d_b_opt, d_a_opt = _precompute_solution(sigma, A, k, gamma, T, steps, Q)
            _solution_cache[key] = (d_b_opt, d_a_opt)
        except Exception:
            # Fallback to simple strategy
            depth = 1.0 / k
            return np.full(len(inventory), depth), np.full(len(inventory), depth)
    
    d_b_opt, d_a_opt = _solution_cache[key]
    
    dt = T / steps
    n = int(t / dt)
    if n >= steps: n = steps - 1
    
    # inventory is array of ints in [-Q, Q]
    # Map to index 0..2Q
    idx = inventory + Q
    
    bid_depths = d_b_opt[n, idx]
    ask_depths = d_a_opt[n, idx]
    
    return bid_depths, ask_depths
