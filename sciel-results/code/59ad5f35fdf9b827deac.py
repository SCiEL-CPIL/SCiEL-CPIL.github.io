import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 260.0
_CACHE = {}

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    
    V = np.zeros((N + 1, n_q))
    opt_b = np.zeros((N, n_q))
    opt_a = np.zeros((N, n_q))
    
    # Grid for depths
    # We use a finer grid to capture the optimal depth more accurately
    max_depth = 5.0 / k
    grid_size = 200
    depths = np.linspace(0, max_depth, grid_size)
    
    # Precompute P(d) for grid
    P_grid = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)
    
    # Precompute terms for vectorization
    # X_base[i] = P_grid[i] * depths[i]
    X_base = P_grid * depths
    # PP[i, j] = P_grid[i] * P_grid[j]
    PP = np.outer(P_grid, P_grid)
    
    for t in range(N - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE - 10:
            # Fallback to seed if too slow
            for qi in range(n_q):
                opt_b[t, qi] = 1.0 / k
                opt_a[t, qi] = 1.0 / k
            return opt_b, opt_a
        
        V_next = V[t + 1, :]
        
        # Calculate deltas for all q
        # delta_plus[qi] = V_next[qi+1] - V_next[qi]
        # delta_minus[qi] = V_next[qi-1] - V_next[qi]
        delta_plus = np.zeros(n_q)
        delta_minus = np.zeros(n_q)
        
        # Interior
        if n_q > 2:
            delta_plus[1:n_q-1] = V_next[2:n_q] - V_next[1:n_q-1]
            delta_minus[1:n_q-1] = V_next[0:n_q-2] - V_next[1:n_q-1]
        
        # Interior: qi in [1, n_q-2]
        if n_q > 2:
            qi_interior = np.arange(1, n_q-1)
            dp = delta_plus[qi_interior]
            dm = delta_minus[qi_interior]
            vn = V_next[qi_interior]
            
            # X[i, q_idx] = P_grid[i]*depths[i] + P_grid[i]*dp[q_idx]
            # X = X_base[:, None] + P_grid[:, None] * dp[None, :]
            X = X_base[:, None] + P_grid[:, None] * dp[None, :]
            
            # Y[j, q_idx] = P_grid[j]*depths[j] + P_grid[j]*dm[q_idx]
            Y = X_base[:, None] + P_grid[:, None] * dm[None, :]
            
            # C[q_idx] = dp[q_idx] + dm[q_idx]
            C = dp + dm
            
            # M[i, j, q_idx] = X[i, q_idx] + Y[j, q_idx] - PP[i, j]*C[q_idx] + vn[q_idx]
            # Broadcast:
            # X is (G, n_int) -> (G, 1, n_int)
            # Y is (G, n_int) -> (1, G, n_int)
            # PP is (G, G) -> (G, G, 1)
            # C is (n_int) -> (1, 1, n_int)
            # vn is (n_int) -> (1, 1, n_int)
            
            M = X[:, None, :] + Y[None, :, :] - PP[:, :, None] * C[None, None, :] + vn[None, None, :]
            
            # Find argmax over i, j for each q
            # Flatten i, j
            flat_M = M.reshape(-1, len(qi_interior))
            flat_max = np.argmax(flat_M, axis=0)
            
            i_star = flat_max // grid_size
            j_star = flat_max % grid_size
            
            V[t, qi_interior] = flat_M[flat_max, np.arange(len(qi_interior))]
            opt_b[t, qi_interior] = depths[i_star]
            opt_a[t, qi_interior] = depths[j_star]
            
        # Boundary q = -Q (qi = 0)
        # Optimize b only, a = 0
        # J(b) = b P(b) + V(-Q) + P(b) delta_plus[0]
        # delta_plus[0] = V(-Q+1) - V(-Q)
        dp0 = V_next[1] - V_next[0]
        # Val[i] = depths[i]*P_grid[i] + V_next[0] + P_grid[i]*dp0
        #        = X_base[i] + P_grid[i]*dp0 + V_next[0]
        val_b = X_base + P_grid * dp0 + V_next[0]
        best_i = np.argmax(val_b)
        V[t, 0] = val_b[best_i]
        opt_b[t, 0] = depths[best_i]
        opt_a[t, 0] = 0.0
        
        # Boundary q = Q (qi = n_q - 1)
        # Optimize a only, b = 0
        # J(a) = a P(a) + V(Q) + P(a) delta_minus[n_q-1]
        # delta_minus[n_q-1] = V(Q-1) - V(Q)
        dm_last = V_next[n_q-2] - V_next[n_q-1]
        val_a = X_base + P_grid * dm_last + V_next[n_q-1]
        best_j = np.argmax(val_a)
        V[t, n_q-1] = val_a[best_j]
        opt_b[t, n_q-1] = 0.0
        opt_a[t, n_q-1] = depths[best_j]
        
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
        
    q_idx = inventory + Q
    
    b_depths = opt_b[t_idx, q_idx]
    a_depths = opt_a[t_idx, q_idx]
    
    return b_depths, a_depths
