import numpy as np

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    N = len(resource_grid)
    T = horizon
    Z = len(income)
    
    # Policy array: c[t, z, i]
    c_policy = np.zeros((T, Z, N))
    
    # Last period: consume everything
    c_policy[T-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    def util_prime(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return 1.0 / c
        else:
            return c ** (-gamma)
            
    def util_double_prime(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return -1.0 / (c * c)
        else:
            return -gamma * c ** (-gamma - 1)

    # Backward induction
    for t in range(T-2, -1, -1):
        c_next = c_policy[t+1]
        
        # Precompute derivatives of the next period's policy for all z'
        dc_dm_next = np.zeros((Z, N))
        for zp in range(Z):
            dc_dm_next[zp] = np.gradient(c_next[zp], resource_grid)
            
        for z in range(Z):
            m_vals = resource_grid
            
            # Initial guess: use the next period's policy for this state z
            # This is a strong prior as policies are smooth and similar across periods
            c_curr = np.clip(c_next[z], 1e-12, m_vals)
            
            # Newton Iterations
            for _ in range(20):
                # Calculate m' for each z'
                # m_prime[z', i] = R * (m[i] - c_curr[i]) + income[z']
                m_prime = R * (m_vals[np.newaxis, :] - c_curr[np.newaxis, :]) + income[:, np.newaxis]
                
                # Interpolate c_next[z'] and its derivative at m_prime
                c_prime = np.zeros((Z, N))
                dc_prime_dm = np.zeros((Z, N))
                
                for zp in range(Z):
                    c_prime[zp] = np.interp(m_prime[zp], resource_grid, c_next[zp])
                    dc_prime_dm[zp] = np.interp(m_prime[zp], resource_grid, dc_dm_next[zp])
                
                # Calculate f(c) and f'(c)
                # f(c) = U'(c) - beta * sum_z' P(z'|z) * R * U'(c')
                up_c = util_prime(c_curr)
                
                exp_term = np.zeros(N)
                exp_term_dd = np.zeros(N)
                
                for zp in range(Z):
                    prob = transition[z, zp]
                    if prob < 1e-15:
                        continue
                    up_cp = util_prime(c_prime[zp])
                    d2up_cp = util_double_prime(c_prime[zp])
                    
                    exp_term += prob * R * up_cp
                    # Derivative term contribution:
                    # d/dc [ R * U'(c') ] = R * U''(c') * (dc'/dc)
                    # dc'/dc = -R * (dc'/dm')
                    # So term is -R^2 * U''(c') * (dc'/dm')
                    exp_term_dd += prob * (-R**2) * d2up_cp * dc_prime_dm[zp]
                
                f_val = up_c - beta * exp_term
                f_prime_val = util_double_prime(c_curr) - beta * exp_term_dd
                
                # Newton step with safeguarding
                # Ensure f_prime_val is not zero
                f_prime_val = np.where(np.abs(f_prime_val) < 1e-10, 1e-10 * np.sign(f_prime_val), f_prime_val)
                f_prime_val = np.where(f_prime_val == 0, 1e-10, f_prime_val)
                
                delta = f_val / f_prime_val
                c_new = c_curr - delta
                
                # Clip to valid range [1e-12, m]
                c_new = np.clip(c_new, 1e-12, m_vals)
                
                # Check convergence
                if np.max(np.abs(c_new - c_curr)) < 1e-10:
                    c_curr = c_new
                    break
                c_curr = c_new
                
            c_policy[t, z, :] = c_curr
            
    return c_policy
