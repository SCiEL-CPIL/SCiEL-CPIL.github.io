import numpy as np

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
    lambda_t = A * dt
    
    # Grid for x
    # x in [1e-8, 1]
    # 1000 points is enough for precision?
    # b = -1/k ln x.
    # If k=1.5, b range 0 to ~12.
    # 1000 points gives resolution ~0.01. Good.
    x_grid = np.logspace(-8, 0, 1000)
    log_x = np.log(x_grid)
    exp_term = np.exp(-lambda_t * x_grid)
    p_grid = 1 - exp_term # shape (1000,)
    
    V = np.zeros((N + 1, 2 * Q + 1))
    opt_b = np.full((N, 2 * Q + 1), np.inf)
    opt_a = np.full((N, 2 * Q + 1), np.inf)
    
    for t in range(N - 1, -1, -1):
        # V[t+1] is shape (2Q+1,)
        # q indices 0..2Q
        # C_b[i] = V[t+1, i+1] - V[t+1, i] for i < 2Q
        # C_b[2Q] = 0 (or irrelevant, q=Q, no bid)
        
        C_b = np.zeros(2 * Q + 1)
        C_a = np.zeros(2 * Q + 1)
        
        # q < Q -> idx < 2Q
        # idx 0..2Q-1
        C_b[:2 * Q] = V[t + 1, 1:] - V[t + 1, :-1]
        
        # q > -Q -> idx > 0
        # idx 1..2Q
        C_a[1:] = V[t + 1, :-1] - V[t + 1, 1:]
        
        # Compute optimal b for all q
        # g_b[i, j] = p_grid[j] * (C_b[i] - 1/k * log_x[j])
        # Shape (2Q+1, 1000)
        
        term_b = C_b[:, None] - (1.0 / k) * log_x[None, :]
        g_b = p_grid[None, :] * term_b
        
        # Find max
        max_idx_b = np.argmax(g_b, axis=1)
        max_val_b = np.max(g_b, axis=1)
        
        # x_opt_b = x_grid[max_idx_b]
        # b_opt_b = -1/k log(x_opt_b)
        
        # If max_val_b <= 0, b = inf
        b_vals = -np.log(x_grid[max_idx_b]) / k
        b_vals[max_val_b <= 1e-9] = np.inf
        
        # Handle boundaries
        # q=Q (idx=2Q): C_b is 0, but we should not quote.
        # In our C_b calc, C_b[2Q] is 0.
        # If C_b=0, optimal b might be 0?
        # g(x) = p (0 - 1/k ln x) = p (-1/k ln x).
        # Max at x=1 (b=0).
        # But we should not quote at q=Q.
        # So we force b=inf for idx=2Q.
        b_vals[2 * Q] = np.inf
        
        opt_b[t, :] = b_vals
        
        # Same for a
        term_a = C_a[:, None] - (1.0 / k) * log_x[None, :]
        g_a = p_grid[None, :] * term_a
        
        max_idx_a = np.argmax(g_a, axis=1)
        max_val_a = np.max(g_a, axis=1)
        
        a_vals = -np.log(x_grid[max_idx_a]) / k
        a_vals[max_val_a <= 1e-9] = np.inf
        
        # q=-Q (idx=0): force a=inf
        a_vals[0] = np.inf
        
        opt_a[t, :] = a_vals
        
        # Update V[t]
        # V[t, i] = V[t+1, i] + cost + max_val_b[i] + max_val_a[i]
        # Note: if b=inf, max_val_b should be 0.
        # Our max_val_b is from g_b. If we set b=inf, the contribution is 0.
        # But max_val_b might be > 0 if the grid found a positive value, but we overrode b to inf?
        # No, we only override b to inf if max_val_b <= 0 OR boundary.
        # If boundary, we should use 0 contribution.
        
        contrib_b = np.where(b_vals == np.inf, 0.0, max_val_b)
        contrib_a = np.where(a_vals == np.inf, 0.0, max_val_a)
        
        q_vals = np.arange(-Q, Q + 1)
        cost = -0.5 * gamma * sigma**2 * q_vals**2 * dt
        
        V[t, :] = V[t + 1, :] + cost + contrib_b + contrib_a

    _cache[key] = (opt_b, opt_a)
    return _cache[key]

def quote(t, inventory, mid, params):
    opt_b, opt_a = _solve_regime(params)
    dt = params['T'] / params['steps']
    Q = params['Q']
    
    t_idx = int(t / dt)
    if t_idx >= len(opt_b):
        t_idx = len(opt_b) - 1
        
    idxs = inventory + Q
    
    bid_depths = opt_b[t_idx, idxs]
    ask_depths = opt_a[t_idx, idxs]
    
    return bid_depths, ask_depths
