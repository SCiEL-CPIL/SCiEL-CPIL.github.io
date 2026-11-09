import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 290.0
_CACHE = {}

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    
    # Value function: V[t, q] for t=0..N, q=-Q..Q
    V = np.zeros((N + 1, n_q))
    opt_b = np.zeros((N, n_q))
    opt_a = np.zeros((N, n_q))
    
    # Grid for depths
    # max_depth: depths beyond 5/k give P < 1-exp(-A*e^{-5}*dt) ~ very small
    # Use a log-spaced grid for better resolution at small depths
    max_depth = 5.0 / k
    grid_size = 250
    # Combine linear and log-spaced for good coverage
    depths_lin = np.linspace(0, max_depth, grid_size // 2 + 1)
    if max_depth > 1e-10:
        depths_log = np.geomspace(max(1e-8, max_depth * 1e-6), max_depth, grid_size // 2)
    else:
        depths_log = np.zeros(grid_size // 2)
    depths = np.unique(np.concatenate([depths_lin, depths_log]))
    depths = np.sort(depths)
    G = len(depths)
    
    # Precompute P(d) for grid
    P_grid = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)
    
    # Precompute terms for vectorization
    # X_base[i] = P_grid[i] * depths[i]
    X_base = P_grid * depths
    # PP[i, j] = P_grid[i] * P_grid[j]
    PP = np.outer(P_grid, P_grid)
    
    # Inventory cost coefficient
    inv_cost = (gamma / 2.0) * sigma**2 * dt
    
    for t in range(N - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE - 15:
            # Fallback: use simple heuristic
            for qi in range(n_q):
                opt_b[t, qi] = 1.0 / k
                opt_a[t, qi] = 1.0 / k
            return opt_b, opt_a
        
        V_next = V[t + 1, :].copy()
        
        # Inventory holding cost: subtract from V_next
        q_vals = np.arange(-Q, Q + 1)
        V_next = V_next - inv_cost * q_vals**2
        
        # Calculate deltas for all q
        # delta_plus[qi] = V_next[qi+1] - V_next[qi]  (ask fill: q -> q+1)
        # delta_minus[qi] = V_next[qi-1] - V_next[qi]  (bid fill: q -> q-1)
        delta_plus = np.zeros(n_q)
        delta_minus = np.zeros(n_q)
        
        if n_q > 2:
            delta_plus[1:n_q-1] = V_next[2:n_q] - V_next[1:n_q-1]
            delta_minus[1:n_q-1] = V_next[0:n_q-2] - V_next[1:n_q-1]
        
        # Interior: qi in [1, n_q-2]
        if n_q > 2:
            qi_interior = np.arange(1, n_q-1)
            dp = delta_plus[qi_interior]
            dm = delta_minus[qi_interior]
            vn = V_next[qi_interior]
            n_int = len(qi_interior)
            
            # X[i, q_idx] = P_grid[i]*depths[i] + P_grid[i]*dp[q_idx]
            # This is the expected value from bid fill: depth earned + change in V
            X = X_base[:, None] + P_grid[:, None] * dp[None, :]
            
            # Y[j, q_idx] = P_grid[j]*depths[j] + P_grid[j]*dm[q_idx]
            Y = X_base[:, None] + P_grid[:, None] * dm[None, :]
            
            # C[q_idx] = dp[q_idx] + dm[q_idx]
            C = dp + dm
            
            # M[i, j, q_idx] = X[i, q_idx] + Y[j, q_idx] - PP[i, j]*C[q_idx] + vn[q_idx]
            M = X[:, None, :] + Y[None, :, :] - PP[:, :, None] * C[None, None, :] + vn[None, None, :]
            
            # Find argmax over i, j for each q
            flat_M = M.reshape(-1, n_int)
            flat_max = np.argmax(flat_M, axis=0)
            
            i_star = flat_max // G
            j_star = flat_max % G
            
            V[t, qi_interior] = flat_M[flat_max, np.arange(n_int)]
            opt_b[t, qi_interior] = depths[i_star]
            opt_a[t, qi_interior] = depths[j_star]
            
        # Boundary q = -Q (qi = 0): no bid fill possible (already at min)
        # Only ask fill possible
        # J(a) = a*P(a) + V_next[-Q] + P(a)*delta_plus[0]
        dp0 = V_next[1] - V_next[0] if n_q > 1 else 0.0
        val_a = X_base + P_grid * dp0 + V_next[0]
        best_j = np.argmax(val_a)
        V[t, 0] = val_a[best_j]
        opt_b[t, 0] = 0.0
        opt_a[t, 0] = depths[best_j]
        
        # Boundary q = Q (qi = n_q-1): no ask fill possible (already at max)
        # Only bid fill possible
        dm_last = V_next[n_q-2] - V_next[n_q-1] if n_q > 1 else 0.0
        val_b = X_base + P_grid * dm_last + V_next[n_q-1]
        best_i = np.argmax(val_b)
        V[t, n_q-1] = val_b[best_i]
        opt_b[t, n_q-1] = depths[best_i]
        opt_a[t, n_q-1] = 0.0
        
    return opt_b, opt_a

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _CACHE:
        _CACHE[key] = _precompute_regime(sigma, A, k, gamma, T, steps, Q)
    
    opt_b, opt_a = _CACHE[key]
    
    t_idx = int(t / (T / steps))
    if t_idx >= steps:
        t_idx = steps - 1
    if t_idx < 0:
        t_idx = 0
    
    q_idx = np.clip(inventory + Q, 0, 2 * Q)
    
    b_depths = opt_b[t_idx, q_idx]
    a_depths = opt_a[t_idx, q_idx]
    
    # Ensure valid depths: no negative, no inf, no NaN
    b_depths = np.where(np.isfinite(b_depths) & (b_depths >= 0), b_depths, 0.0)
    a_depths = np.where(np.isfinite(a_depths) & (a_depths >= 0), a_depths, 0.0)
    
    return b_depths, a_depths
