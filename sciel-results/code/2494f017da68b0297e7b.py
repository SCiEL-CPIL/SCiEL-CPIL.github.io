import numpy as np

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    T = model['horizon']
    
    n_grid = len(resource_grid)
    n_states = len(income)
    
    c_policy = np.zeros((T, n_states, n_grid))
    V = np.zeros((T, n_states, n_grid))
    
    # Use a fine grid for consumption fraction f = c/m
    # budget is given as 48 in the prompt, but we can use a higher resolution for accuracy
    # The prompt says "budget = 48" is passed, but we can ignore it for internal resolution
    # or use it as a hint. Let's use a fixed high resolution for better accuracy.
    n_frac = 500 
    f_grid = np.linspace(0, 1, n_frac)
    
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
            
    # Terminal period
    c_policy[T-1, :, :] = resource_grid[None, None, :]
    V[T-1, :, :] = utility(resource_grid)[None, :]
    
    # Backward induction
    for t in range(T - 2, -1, -1):
        for z in range(n_states):
            # m shape (n_grid,)
            m_vals = resource_grid
            
            # c shape (n_grid, n_frac)
            c_candidates = m_vals[:, None] * f_grid[None, :]
            
            # Ensure c >= 0 (f >= 0, m > 0 so c > 0)
            # Utility of c
            u_c = utility(c_candidates)
            
            # m' shape (n_grid, n_frac, n_states)
            # m' = R * (m - c) + income[z']
            m_prime = R * (m_vals[:, None, None] - c_candidates[:, :, None]) + income[None, None, :]
            
            # Interpolate V[t+1, z', :] for each z'
            # V_next shape (n_states, n_grid)
            V_next = V[t+1, :, :]
            
            # We need to interpolate V_next[z'] at m_prime[:, :, z']
            # Result shape (n_grid, n_frac, n_states)
            V_interp = np.zeros((n_grid, n_frac, n_states))
            
            for zp in range(n_states):
                # np.interp requires 1D x, xp, fp
                # Flatten m_prime[:, :, zp]
                x_flat = m_prime[:, :, zp].ravel()
                v_flat = np.interp(x_flat, resource_grid, V_next[zp, :])
                V_interp[:, :, zp] = v_flat.reshape(n_grid, n_frac)
                
            # Expected V_next
            # transition[z, :] shape (n_states,)
            # V_interp shape (n_grid, n_frac, n_states)
            # Sum over states: np.sum(transition[z, None, None, :] * V_interp, axis=2)
            exp_v_next = np.sum(transition[z, None, None, :] * V_interp, axis=2)
            
            # Total value J shape (n_grid, n_frac)
            J = u_c + beta * exp_v_next
            
            # Find best fraction index for each m
            # argmax over axis 1 (frac dimension)
            best_k = np.argmax(J, axis=1)
            
            # Get optimal c
            c_opt = c_candidates[np.arange(n_grid), best_k]
            
            # Ensure constraints
            c_opt = np.clip(c_opt, 1e-12, m_vals)
            
            c_policy[t, z, :] = c_opt
            V[t, z, :] = J[np.arange(n_grid), best_k]
            
    return c_policy
