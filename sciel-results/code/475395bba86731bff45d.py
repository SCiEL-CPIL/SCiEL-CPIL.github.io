import numpy as np

_cache = {}

def _compute_tables(sigma, A, k, gamma, T, steps, Q):
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _cache:
        return _cache[key]
    
    dt = T / steps
    lambda_ = A * dt
    
    # Grid
    x_grid = np.linspace(0, 20, 100)
    u_grid = np.exp(-x_grid)
    p_grid = 1 - np.exp(-lambda_ * u_grid)
    d_grid = x_grid / k
    
    # V_next
    V_next = np.zeros(2 * Q + 1)
    
    D_b = np.full((steps, 2 * Q + 1), np.inf)
    D_a = np.full((steps, 2 * Q + 1), np.inf)
    
    # Precompute indices
    idxs = np.arange(2 * Q + 1)
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(2 * Q + 1)
        
        # Calculate C_plus and C_minus for all idx
        # C_plus[idx] = V_next[idx+1] - V_next[idx]
        # Valid for idx < 2Q
        C_plus = np.zeros(2 * Q + 1)
        C_minus = np.zeros(2 * Q + 1)
        
        # idx 0 to 2Q-1
        C_plus[:2*Q] = V_next[1:] - V_next[:-1]
        # idx 2Q (q=Q)
        C_plus[2*Q] = -np.inf 
        
        # idx 1 to 2Q
        C_minus[1:] = V_next[:-1] - V_next[1:]
        # idx 0 (q=-Q)
        C_minus[0] = -np.inf
        
        # Compute H_b: (2Q+1, M)
        # H_b[i, j] = p_grid[j] * (d_grid[j] + C_plus[i])
        # Use broadcasting
        # C_plus shape (2Q+1, 1)
        # d_grid shape (1, M)
        H_b = p_grid[None, :] * (d_grid[None, :] + C_plus[:, None])
        # For indices where C_plus is -inf, H_b is -inf.
        # We want max. If all -inf, argmax is 0?
        # We handle boundaries separately or mask.
        
        # Mask for valid bid quotes (q < Q => idx < 2Q)
        mask_b = C_plus > -np.inf
        
        # For valid ones, find argmax
        # For invalid, set to inf later
        
        # To avoid issues with -inf, we can set H_b to -np.inf where invalid
        H_b[~mask_b, :] = -np.inf
        
        max_idx_b = np.argmax(H_b, axis=1)
        max_val_b = np.take_along_axis(H_b, max_idx_b[:, None], axis=1).flatten()
        
        # Where max_val_b <= 0, we set depth to inf
        valid_b = max_val_b > 0
        
        # Extract depths
        depths_b = d_grid[max_idx_b]
        D_b[n, :] = np.where(valid_b, depths_b, np.inf)
        
        # Similarly for ask
        H_a = p_grid[None, :] * (d_grid[None, :] + C_minus[:, None])
        mask_a = C_minus > -np.inf
        H_a[~mask_a, :] = -np.inf
        
        max_idx_a = np.argmax(H_a, axis=1)
        max_val_a = np.take_along_axis(H_a, max_idx_a[:, None], axis=1).flatten()
        
        valid_a = max_val_a > 0
        
        depths_a = d_grid[max_idx_a]
        D_a[n, :] = np.where(valid_a, depths_a, np.inf)
        
        # Update V
        # V_curr = max_val_b + max_val_a + V_next - cost
        # max_val_b is -inf where invalid, but we want 0 contribution there?
        # If invalid (boundary), contribution is 0.
        # So use np.where(valid_b, max_val_b, 0)
        
        contrib_b = np.where(valid_b, max_val_b, 0.0)
        contrib_a = np.where(valid_a, max_val_a, 0.0)
        
        q_vals = idxs - Q
        cost = 0.5 * gamma * sigma**2 * q_vals**2 * dt
        
        V_curr = contrib_b + contrib_a + V_next - cost
        
        V_next = V_curr
    
    _cache[key] = (D_b, D_a)
    return D_b, D_a

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    n = int(round(t / (T / steps)))
    n = np.clip(n, 0, steps - 1)
    
    D_b, D_a = _compute_tables(sigma, A, k, gamma, T, steps, Q)
    
    inv_idx = np.asarray(inventory, dtype=np.int32) + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    bid_depths = D_b[n, inv_idx]
    ask_depths = D_a[n, inv_idx]
    
    return bid_depths, ask_depths
