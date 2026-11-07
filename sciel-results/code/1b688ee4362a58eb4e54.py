"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin below 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    n_z = len(income)
    
    # Use a dense grid for internal computation
    m_grid = np.geomspace(0.001, 40, 251)
    n_m = len(m_grid)
    
    # Utility function
    def u(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c_safe)
        else:
            return c_safe ** (1.0 - gamma) / (1.0 - gamma)
    
    # Policy: c[t, z, m] for t in 0..horizon-1, z in 0..n_z-1, m in m_grid
    c_policy = np.zeros((horizon, n_z, n_m))
    
    # Terminal condition: c = m
    for z in range(n_z):
        c_policy[horizon - 1, z, :] = m_grid
    
    # Backward induction from t = horizon-2 down to 0
    n_c = 301  # Number of consumption fractions to evaluate
    fracs = np.linspace(0.001, 1.0, n_c)
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            # Not enough time, break and return what we have
            break
        
        # Compute V_{t+1}(m, z') for each z'
        V_next = np.zeros((n_z, n_m))
        
        if t == horizon - 2:
            # V_{t+1} = V_{horizon-1} = u(m) since terminal
            for zp in range(n_z):
                V_next[zp, :] = u(m_grid)
        else:
            # Compute V_{t+1} for each z'
            for zp in range(n_z):
                c_next = c_policy[t + 1, zp, :]
                u_next = u(c_next)
                
                # Add beta * sum_{z''} transition[zp, z''] * V_{t+2}(m'', z'')
                cont_sum = np.zeros(n_m)
                for zpp in range(n_z):
                    prob = transition[zp, zpp]
                    if prob < 1e-15:
                        continue
                    m_next = R * (m_grid - c_next) + income[zpp]
                    m_next = np.clip(m_next, 1e-12, 40.0)
                    
                    # Compute V_{t+2} on the fly for each z''
                    V_next2 = np.zeros(n_m)
                    for zpp2 in range(n_z):
                        prob2 = transition[zpp, zpp2]
                        if prob2 < 1e-15:
                            continue
                        c_pp = c_policy[t + 2, zpp2, :]
                        u_pp = u(c_pp)
                        m_pp = R * (m_grid - c_pp) + income[zpp2]
                        m_pp = np.clip(m_pp, 1e-12, 40.0)
                        
                        if t + 3 == horizon - 1:
                            V_pp = u(m_grid)
                        else:
                            # Compute V_{t+3} recursively
                            V_pp = np.zeros(n_m)
                            for zpp3 in range(n_z):
                                prob3 = transition[zpp2, zpp3]
                                if prob3 < 1e-15:
                                    continue
                                c_ppp = c_policy[t + 3, zpp3, :]
                                u_ppp = u(c_ppp)
                                m_ppp = R * (m_grid - c_ppp) + income[zpp3]
                                m_ppp = np.clip(m_ppp, 1e-12, 40.0)
                                
                                if t + 4 == horizon - 1:
                                    V_ppp = u(m_grid)
                                else:
                                    # Deep recursion - approximate
                                    V_ppp = u(np.minimum(m_ppp, m_grid[-1]))
                                
                                V_pp += prob3 * np.interp(m_ppp, m_grid, V_ppp)
                            V_pp = u(c_ppp) + beta * V_pp
                        
                        V_next2 += prob2 * np.interp(m_pp, m_grid, V_pp)
                    
                    V_next2 = u(c_pp) + beta * V_next2
                    cont_sum += prob * np.interp(m_next, m_grid, V_next2)
                
                V_next[zp, :] = u_next + beta * cont_sum
        
        # Now for each z, find optimal c for each m using vectorized grid search
        for z in range(n_z):
            best_c = np.zeros(n_m)
            
            for i in range(n_m):
                m_i = m_grid[i]
                c_vals = m_i * fracs
                c_vals = np.clip(c_vals, 1e-12, m_i)
                
                # Compute u(c) for all c_vals
                u_vals = u(c_vals)
                
                # Compute continuation value for each c
                total_cont = np.zeros(n_c)
                for zp in range(n_z):
                    prob = transition[z, zp]
                    if prob < 1e-15:
                        continue
                    m_prime = R * (m_i - c_vals) + income[zp]
                    m_prime = np.clip(m_prime, 1e-12, 40.0)
                    v_interp = np.interp(m_prime, m_grid, V_next[zp, :])
                    total_cont += prob * v_interp
                
                obj = u_vals + beta * total_cont
                best_idx = np.argmax(obj)
                best_c[i] = c_vals[best_idx]
            
            c_policy[t, z, :] = best_c
    
    # Interpolate back to the original resource_grid
    result = np.zeros((horizon, n_z, len(resource_grid)))
    for t in range(horizon):
        for z in range(n_z):
            result[t, z, :] = np.interp(resource_grid, m_grid, c_policy[t, z, :])
    
    # Enforce constraints: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            result[t, z, :] = np.maximum(result[t, z, :], 1e-12)
            result[t, z, :] = np.minimum(result[t, z, :], resource_grid + 1e-10)
    
    # Terminal condition: c[horizon-1] must equal the grid within 1e-9
    for z in range(n_z):
        result[horizon - 1, z, :] = resource_grid
    
    # Check finiteness
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=1e-12, posinf=1.0, neginf=1e-12)
        for t in range(horizon):
            for z in range(n_z):
                result[t, z, :] = np.maximum(result[t, z, :], 1e-12)
                result[t, z, :] = np.minimum(result[t, z, :], resource_grid + 1e-10)
        for z in range(n_z):
            result[horizon - 1, z, :] = resource_grid
    
    return result
# EVOLVE-BLOCK-END
