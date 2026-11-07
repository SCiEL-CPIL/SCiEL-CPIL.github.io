import numpy as np

DP_CACHE = {}

def _compute_dp(sigma, A, k, gamma, T, steps, Q):
    """
    Precomputes the optimal policy (depths) and value function for a given regime.
    Returns D_b, D_a arrays of shape (steps, 2*Q+1).
    """
    N = steps
    Q_range = 2 * Q + 1
    V = np.zeros((N + 1, Q_range))
    D_b = np.zeros((N, Q_range))
    D_a = np.zeros((N, Q_range))
    
    dt = T / N
    
    # Grid for optimization
    # d in [0, 5/k]
    d_max = 5.0 / k
    grid_size = 100
    d_grid = np.linspace(0, d_max, grid_size)
    
    # Precompute p for grid
    # p = 1 - exp(-A * exp(-k * d) * dt)
    exp_term = np.exp(-k * d_grid)
    p_grid = 1.0 - np.exp(-A * exp_term * dt)
    
    # Precompute reward terms for grid
    R_grid = p_grid * d_grid
    
    q_vals = np.arange(-Q, Q + 1)
    # Penalty: gamma / 2 * sigma^2 * q^2 * dt
    penalty = (gamma / 2.0) * (sigma ** 2) * (q_vals ** 2) * dt
    
    for n in range(N - 1, -1, -1):
        V_next = V[n + 1]
        
        # J shape: (Q_range, grid_size, grid_size)
        # We compute for each q
        J_flat = np.zeros((Q_range, grid_size * grid_size))
        
        for q_idx in range(Q_range):
            q_val = q_vals[q_idx]
            
            V_q = V_next[q_idx]
            
            # V for q+1 (bid fill)
            if q_val < Q:
                V_q_plus = V_next[q_idx + 1]
            else:
                V_q_plus = -np.inf
            
            # V for q-1 (ask fill)
            if q_val > -Q:
                V_q_minus = V_next[q_idx - 1]
            else:
                V_q_minus = -np.inf

            # Coefficients for Future Value
            # p_b is (G,), p_a is (G,)
            one_minus_pb = 1.0 - p_grid
            one_minus_pa = 1.0 - p_grid
            
            # Term V_q: (1-p_b)(1-p_a) + p_b p_a
            coeff_q = (one_minus_pb[:, None] * one_minus_pa[None, :]) + (p_grid[:, None] * p_grid[None, :])
            
            # Term V_q_plus: p_b (1-p_a)
            coeff_q_plus = p_grid[:, None] * one_minus_pa[None, :]
            
            # Term V_q_minus: (1-p_b) p_a
            coeff_q_minus = one_minus_pb[:, None] * p_grid[None, :]
            
            # If V is -inf, the product should be 0 if coeff is 0, else -inf
            # Using np.where or just letting it propagate?
            # If V_q_plus is -inf and coeff > 0, EV is -inf. This is correct.
            # If coeff is 0, 0 * -inf is NaN.
            # We need to handle 0 * -inf = 0.
            
            # Safe multiplication
            EV = np.zeros((grid_size, grid_size))
            
            # Add V_q term
            EV += V_q * coeff_q
            
            # Add V_q_plus term
            if V_q_plus > -np.inf / 2: # Check if valid
                 EV += V_q_plus * coeff_q_plus
            else:
                 # If V_q_plus is -inf, we only add if coeff is 0? No, if coeff > 0, it's -inf.
                 # So we can just multiply, but handle NaNs.
                 term = V_q_plus * coeff_q_plus
                 term = np.where(coeff_q_plus == 0, 0, term)
                 EV += term

            # Add V_q_minus term
            if V_q_minus > -np.inf / 2:
                 EV += V_q_minus * coeff_q_minus
            else:
                 term = V_q_minus * coeff_q_minus
                 term = np.where(coeff_q_minus == 0, 0, term)
                 EV += term

            # Immediate Reward: R_b[i] + R_a[j]
            # R_grid is (G,)
            Reward = R_grid[:, None] + R_grid[None, :]
            
            # Total J
            J_q = Reward + EV - penalty
            
            J_flat[q_idx] = J_q.flatten()
            
        # Find max
        max_indices = np.argmax(J_flat, axis=1)
        
        max_i = max_indices // grid_size
        max_j = max_indices % grid_size
        
        D_b[n] = d_grid[max_i]
        D_a[n] = d_grid[max_j]
        
        V[n] = J_flat[np.arange(Q_range), max_indices]
        
        # Enforce boundaries
        D_b[n, 2 * Q] = np.inf
        D_a[n, 0] = np.inf
        
    return D_b, D_a

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in DP_CACHE:
        DP_CACHE[key] = _compute_dp(sigma, A, k, gamma, T, steps, Q)
    
    D_b, D_a = DP_CACHE[key]
    
    n = int(t / (T / steps))
    if n >= steps:
        n = steps - 1
        
    q_indices = inventory + Q
    
    bid_depths = D_b[n, q_indices]
    ask_depths = D_a[n, q_indices]
    
    return bid_depths, ask_depths
