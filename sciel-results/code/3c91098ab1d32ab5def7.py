import numpy as np
import time
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    M = len(resource_grid)
    Z = len(income)
    
    # Initialize consumption array
    # Shape: (horizon, Z, M)
    consumption = np.zeros((horizon, Z, M))
    
    # Value function V[t, z, m_index]
    V = np.zeros((horizon, Z, M))
    
    def utility(c):
        # c is scalar or array
        c_safe = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c_safe)
        else:
            return (c_safe ** (1 - gamma)) / (1 - gamma)

    # Initialize V for t = horizon - 1 (index 5)
    # V[5, z, i] = U(m_i)
    V[horizon - 1, :, :] = utility(resource_grid)[np.newaxis, np.newaxis, :]
    consumption[horizon - 1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    # Grid for consumption fraction s = c/m
    # We want to cover the range well.
    S = 1000
    # Avoid 0.
    s_grid = np.linspace(1e-4, 1.0, S)
    
    # Backward induction for t = horizon - 2 down to 0
    for t in range(horizon - 2, -1, -1):
        # Check time
        if time.time() > deadline:
            # If we run out of time, we must return valid consumption.
            # Since we go backwards, if we fail at t, we haven't computed 0..t.
            # We can fill uncomputed t with a heuristic (e.g., c=m) to ensure validity?
            # But the contract says c[5] must be grid.
            # If we break, consumption[0..t] is 0. 0 is invalid (0 < c).
            # So we should ensure we finish. The vectorized approach is very fast.
            pass

        V_next = V[t + 1] # Shape (Z, M)
        
        for z in range(Z):
            # Vectorized coarse search
            m_col = resource_grid[:, np.newaxis]
            s_row = s_grid[np.newaxis, :]
            
            c_candidates = m_col * s_row
            c_candidates = np.maximum(c_candidates, 1e-12)
            
            obj_values = np.zeros((M, S))
            
            for zp in range(Z):
                prob = transition[z, zp]
                if prob == 0:
                    continue
                
                m_prime = R * (resource_grid[:, np.newaxis] - c_candidates) + income[zp]
                
                # Interpolate V_next[zp] at m_prime
                m_prime_flat = m_prime.flatten()
                v_interp = np.interp(m_prime_flat, resource_grid, V_next[zp, :])
                v_interp = v_interp.reshape(M, S)
                
                obj_values += prob * v_interp
            
            u_c = utility(c_candidates)
            total_obj = u_c + beta * obj_values
            
            best_idx = np.argmax(total_obj, axis=1)
            best_s = s_grid[best_idx]
            best_c = resource_grid * best_s
            
            # Refinement
            for i in range(M):
                m_i = resource_grid[i]
                c_init = best_c[i]
                
                # Define bounds for local search
                if S > 1:
                    s_step = s_grid[1] - s_grid[0]
                else:
                    s_step = 1.0
                
                c_low = max(1e-12, m_i * (best_s[i] - 2*s_step))
                c_high = min(m_i, m_i * (best_s[i] + 2*s_step))
                
                if c_high - c_low < 1e-12:
                    c_opt = c_init
                    V[t, z, i] = total_obj[i, best_idx[i]]
                    consumption[t, z, i] = c_opt
                    continue

                def neg_obj(c_val):
                    c_val = np.clip(c_val, 1e-12, m_i)
                    m_save = R * (m_i - c_val)
                    
                    cont_val = 0.0
                    for zp in range(Z):
                        prob = transition[z, zp]
                        if prob == 0:
                            continue
                        m_prime = m_save + income[zp]
                        v_zp = np.interp(m_prime, resource_grid, V_next[zp, :])
                        cont_val += prob * v_zp
                        
                    obj = utility(c_val) + beta * cont_val
                    return -obj

                res = minimize_scalar(neg_obj, bounds=(c_low, c_high), method='bounded')
                c_opt = res.x
                c_opt = np.clip(c_opt, 1e-12, m_i)
                
                consumption[t, z, i] = c_opt
                V[t, z, i] = -res.fun

    return consumption
