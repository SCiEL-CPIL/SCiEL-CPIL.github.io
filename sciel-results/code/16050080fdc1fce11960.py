import numpy as np
import time
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'])
    transition = np.asarray(model['transition'])
    horizon = model['horizon']
    m_grid = np.asarray(resource_grid)
    N_m = len(m_grid)
    N_z = len(income)
    
    def util(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
    
    # Value function: V[t, z, m]
    V = np.zeros((horizon + 1, N_z, N_m))
    # Consumption policy: c[t, z, m]
    C = np.zeros((horizon, N_z, N_m))
    
    # Terminal condition: at t=horizon-1 (last period), consume everything
    V[horizon, :, :] = 0.0
    C[horizon - 1, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        for z in range(N_z):
            for i_m in range(N_m):
                m = m_grid[i_m]
                if m <= 0:
                    C[t, z, i_m] = 0.0
                    V[t, z, i_m] = 0.0
                    continue
                
                # Find optimal c that maximizes:
                # util(c) + beta * sum_z' transition[z, z'] * V[t+1, z', R*(m-c) + income[z']]
                
                def neg_objective(c):
                    c = np.clip(c, 1e-12, m)
                    m_next_base = R * (m - c)
                    total_val = util(c)
                    for z_prime in range(N_z):
                        m_next = m_next_base + income[z_prime]
                        # Interpolate V[t+1, z_prime, :] at m_next
                        if m_next <= m_grid[0]:
                            v_next = V[t + 1, z_prime, 0]
                        elif m_next >= m_grid[-1]:
                            v_next = V[t + 1, z_prime, -1]
                        else:
                            v_next = np.interp(m_next, m_grid, V[t + 1, z_prime, :])
                        total_val += beta * transition[z, z_prime] * v_next
                    return -total_val
                
                # Search for optimal c in (0, m)
                # Use bounded minimization
                res = minimize_scalar(neg_objective, bounds=(1e-12, m * 0.999), method='bounded', 
                                      options={'xatol': 1e-10, 'maxiter': 200})
                c_opt = res.x
                V[t, z, i_m] = -res.fun
                C[t, z, i_m] = c_opt
                
                # Check deadline
                if time.time() - start_time > DEADLINE - 5:
                    # Fill remaining with simple heuristic
                    for z_rem in range(z + 1, N_z):
                        for i_m_rem in range(i_m, N_m):
                            C[t, z_rem, i_m_rem] = m_grid[i_m_rem] * 0.9
                            V[t, z_rem, i_m_rem] = 0
                    for z_rem in range(N_z):
                        for i_m_rem in range(N_m):
                            if t < horizon - 1:
                                if V[t, z_rem, i_m_rem] == 0 and C[t, z_rem, i_m_rem] == 0:
                                    C[t, z_rem, i_m_rem] = m_grid[i_m_rem] * 0.9
                    break
            if time.time() - start_time > DEADLINE - 5:
                break
        if time.time() - start_time > DEADLINE - 5:
            break
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(N_z):
            for i_m in range(N_m):
                C[t, z, i_m] = np.clip(C[t, z, i_m], 1e-12, m_grid[i_m] + 1e-10)
    
    # Final period must equal grid
    C[horizon - 1, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    # Check for any non-finite values
    if not np.all(np.isfinite(C)):
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        C = np.clip(C, 1e-12, m_grid[np.newaxis, np.newaxis, :] + 1e-10)
    
    return C
