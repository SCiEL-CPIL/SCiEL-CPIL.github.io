import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 290.0
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
    
    # Grid for depth b (and a)
    # b = -1/k * ln(x), x in (0, 1]
    # Use a fine grid in x-space
    x_grid = np.logspace(-14, 0, 8000)
    b_grid = -np.log(x_grid) / k
    p_grid = 1.0 - np.exp(-A * x_grid * dt)
    
    V = np.zeros((N + 1, n_q))
    opt_b = np.full((N, n_q), np.inf)
    opt_a = np.full((N, n_q), np.inf)
    
    # Pre-compute cost array
    cost_arr = -0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    for t in range(N - 1, -1, -1):
        if time.time() > _DEADLINE - 10:
            # Fallback: fill remaining with heuristic
            for tt in range(t, -1, -1):
                for i in range(n_q):
                    if i < 2 * Q:  # q < Q
                        opt_b[tt, i] = 1.0 / k
                    else:
                        opt_b[tt, i] = np.inf
                    if i > 0:  # q > -Q
                        opt_a[tt, i] = 1.0 / k
                    else:
                        opt_a[tt, i] = np.inf
            break
        
        V_next = V[t + 1]
        
        # D_b[i] = V[t+1, i+1] - V[t+1, i]  (for i = 0..n_q-2)
        # D_a[i] = V[t+1, i-1] - V[t+1, i]  (for i = 1..n_q-1)
        # D_cross[i] = V[t+1, i+1] - 2*V[t+1, i] + V[t+1, i-1]
        
        D_b = np.zeros(n_q)
        D_a = np.zeros(n_q)
        D_cross = np.zeros(n_q)
        
        D_b[:-1] = V_next[1:] - V_next[:-1]
        D_a[1:] = V_next[:-1] - V_next[1:]
        
        # D_cross is defined for i = 1..n_q-2
        if n_q > 2:
            D_cross[1:-1] = V_next[2:] - 2.0 * V_next[1:-1] + V_next[:-2]
        
        # Decoupled optimization (first-order approximation)
        # For bid: maximize p_j * (b_j + D_b[i])
        # term_b[i, j] = D_b[i] + b_grid[j]
        term_b = D_b[:, None] + b_grid[None, :]
        val_b = p_grid[None, :] * term_b
        
        max_idx_b = np.argmax(val_b, axis=1)
        max_val_b = np.max(val_b, axis=1)
        
        # For ask: maximize p_j * (a_j + D_a[i])
        term_a = D_a[:, None] + b_grid[None, :]
        val_a = p_grid[None, :] * term_a
        
        max_idx_a = np.argmax(val_a, axis=1)
        max_val_a = np.max(val_a, axis=1)
        
        b_vals = b_grid[max_idx_b]
        a_vals = b_grid[max_idx_a]
        
        # Determine no-quote conditions
        no_quote_b = max_val_b < 1e-10
        no_quote_a = max_val_a < 1e-10
        
        # Boundary conditions
        no_quote_b[2 * Q] = True  # q = Q, no bid
        no_quote_a[0] = True      # q = -Q, no ask
        
        b_vals[no_quote_b] = np.inf
        a_vals[no_quote_a] = np.inf
        
        opt_b[t, :] = b_vals
        opt_a[t, :] = a_vals
        
        contrib_b = np.where(no_quote_b, 0.0, max_val_b)
        contrib_a = np.where(no_quote_a, 0.0, max_val_a)
        
        V[t, :] = V_next + cost_arr + contrib_b + contrib_a
    
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
    
    # Ensure validity
    bid_depths[~np.isfinite(bid_depths)] = np.inf
    ask_depths[~np.isfinite(ask_depths)] = np.inf
    
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    return bid_depths, ask_depths
