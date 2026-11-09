import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 295.0
_cache = {}

def _solve_regime(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    
    # Use a finer grid for higher precision
    x_grid = np.logspace(-12, 0, 10000)
    log_x = np.log(x_grid)
    exp_term = np.exp(-A * dt * x_grid)
    p_grid = 1 - exp_term
    inv_k = 1.0 / k
    
    V = np.zeros((N + 1, n_q))
    opt_b = np.full((N, n_q), np.inf)
    opt_a = np.full((N, n_q), np.inf)
    
    for t in range(N - 1, -1, -1):
        if time.time() > _DEADLINE - 5:
            # Fallback to a simpler heuristic if time runs out
            for tt in range(t, -1, -1):
                for i in range(n_q):
                    q = q_vals[i]
                    if q < Q:
                        opt_b[tt, i] = 1.0 / k
                    if q > -Q:
                        opt_a[tt, i] = 1.0 / k
            break
            
        C_b = np.zeros(n_q)
        C_a = np.zeros(n_q)
        
        C_b[:-1] = V[t + 1, 1:] - V[t + 1, :-1]
        C_a[1:] = V[t + 1, :-1] - V[t + 1, 1:]
        
        term_b = C_b[:, None] - inv_k * log_x[None, :]
        g_b = p_grid[None, :] * term_b
        
        max_idx_b = np.argmax(g_b, axis=1)
        max_val_b = np.max(g_b, axis=1)
        
        b_vals = -np.log(x_grid[max_idx_b]) * inv_k
        b_vals[max_val_b <= 1e-12] = np.inf
        b_vals[2 * Q] = np.inf
        
        opt_b[t, :] = b_vals
        
        term_a = C_a[:, None] - inv_k * log_x[None, :]
        g_a = p_grid[None, :] * term_a
        
        max_idx_a = np.argmax(g_a, axis=1)
        max_val_a = np.max(g_a, axis=1)
        
        a_vals = -np.log(x_grid[max_idx_a]) * inv_k
        a_vals[max_val_a <= 1e-12] = np.inf
        a_vals[0] = np.inf
        
        opt_a[t, :] = a_vals
        
        contrib_b = np.where(b_vals == np.inf, 0.0, max_val_b)
        contrib_a = np.where(a_vals == np.inf, 0.0, max_val_a)
        
        cost = -0.5 * gamma * sigma**2 * q_vals**2 * dt
        
        V[t, :] = V[t + 1, :] + cost + contrib_b + contrib_a
        
        # Refinement step: local search around grid-optimal points
        # Only refine if we have time to spare
        if time.time() < _DEADLINE - 10:
            # Refine b_vals
            for i in range(n_q):
                if not np.isinf(b_vals[i]):
                    idx = max_idx_b[i]
                    best_idx = idx
                    best_val = max_val_b[i]
                    for d in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
                        ni = idx + d
                        if 0 <= ni < len(x_grid):
                            val = p_grid[ni] * (C_b[i] - inv_k * log_x[ni])
                            if val > best_val:
                                best_val = val
                                best_idx = ni
                    if best_idx != idx:
                        b_vals[i] = -np.log(x_grid[best_idx]) * inv_k
                        max_val_b[i] = best_val
                        max_idx_b[i] = best_idx
                
                # Refine a_vals
                if not np.isinf(a_vals[i]):
                    idx = max_idx_a[i]
                    best_idx = idx
                    best_val = max_val_a[i]
                    for d in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
                        ni = idx + d
                        if 0 <= ni < len(x_grid):
                            val = p_grid[ni] * (C_a[i] - inv_k * log_x[ni])
                            if val > best_val:
                                best_val = val
                                best_idx = ni
                    if best_idx != idx:
                        a_vals[i] = -np.log(x_grid[best_idx]) * inv_k
                        max_val_a[i] = best_val
                        max_idx_a[i] = best_idx
            
            # Update contributions after refinement
            contrib_b = np.where(b_vals == np.inf, 0.0, max_val_b)
            contrib_a = np.where(a_vals == np.inf, 0.0, max_val_a)
            V[t, :] = V[t + 1, :] + cost + contrib_b + contrib_a

    _cache[key] = (opt_b, opt_a)
    return _cache[key]

def quote(t, inventory, mid, params):
    if time.time() > _DEADLINE:
        return np.full(4000, np.inf), np.full(4000, np.inf)
        
    opt_b, opt_a = _solve_regime(params)
    dt = params['T'] / params['steps']
    Q = params['Q']
    
    t_idx = int(t / dt)
    if t_idx >= len(opt_b):
        t_idx = len(opt_b) - 1
        
    idxs = inventory + Q
    
    bid_depths = opt_b[t_idx, idxs].copy()
    ask_depths = opt_a[t_idx, idxs].copy()
    
    # Replace any non-finite values that are not inf with inf
    bid_depths[~np.isfinite(bid_depths)] = np.inf
    ask_depths[~np.isfinite(ask_depths)] = np.inf
    
    # Ensure no negative depths
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    return bid_depths, ask_depths
