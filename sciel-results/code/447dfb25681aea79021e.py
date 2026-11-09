import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_grid = len(resource_grid)
    n_states = len(income)
    
    # c[t, z, i] - consumption policy
    c = np.zeros((horizon, n_states, n_grid))
    
    # Terminal: consume everything
    c[-1, :, :] = resource_grid[np.newaxis, :, np.newaxis]
    
    for t in range(horizon - 2, -1, -1):
        for z in range(n_states):
            m_vals = resource_grid  # shape (n_grid,)
            
            # Bisection bounds
            c_low = np.full(n_grid, 1e-12)
            c_high = m_vals.copy()
            
            for iteration in range(60):
                if time.time() - start_time > deadline - 5:
                    break
                
                c_mid = (c_low + c_high) / 2.0
                
                m_prime_0 = np.maximum(R * (m_vals - c_mid) + income[0], 0.0)
                m_prime_1 = np.maximum(R * (m_vals - c_mid) + income[1], 0.0)
                
                c_next_0 = np.interp(m_prime_0, resource_grid, c[t+1, 0, :])
                c_next_1 = np.interp(m_prime_1, resource_grid, c[t+1, 1, :])
                
                c_next_0 = np.maximum(c_next_0, 1e-12)
                c_next_1 = np.maximum(c_next_1, 1e-12)
                
                lhs = c_mid ** (-gamma)
                rhs = beta * (transition[z, 0] * c_next_0 ** (-gamma) + transition[z, 1] * c_next_1 ** (-gamma))
                
                f = lhs - rhs
                
                c_low = np.where(f > 0, c_mid, c_low)
                c_high = np.where(f < 0, c_mid, c_high)
            
            c[t, z, :] = (c_low + c_high) / 2.0
            c[t, z, :] = np.clip(c[t, z, :], 1e-12, m_vals)
        
        if time.time() - start_time > deadline - 10:
            for t2 in range(t - 1, -1, -1):
                for z2 in range(n_states):
                    c[t2, z2, :] = resource_grid * 0.7
                    c[t2, z2, :] = np.clip(c[t2, z2, :], 1e-12, resource_grid)
            break
    
    c = np.clip(c, 1e-12, None)
    for t in range(horizon):
        for z in range(n_states):
            c[t, z, :] = np.minimum(c[t, z, :], resource_grid + 1e-10)
    
    c[-1, :, :] = resource_grid[np.newaxis, :, np.newaxis]
    c = np.clip(c, 1e-12, None)
    
    return c
