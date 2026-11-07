"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin below 120s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    def utility(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return c**(1 - gamma) / (1 - gamma)
    
    # Use the scoring grid for internal computation
    m_grid = np.geomspace(0.001, 40, 251)
    n_m = len(m_grid)
    n_z = len(income)
    
    # Final period: consume all
    c_final = np.tile(m_grid, (n_z, 1))  # shape (n_z, n_m)
    
    # Backward induction
    # V[t+1][z][m] = value function at t+1
    V_next = np.zeros((n_z, n_m))  # value at t+1 for each z, m
    
    c_policies = [None] * horizon
    c_policies[horizon - 1] = c_final
    
    for t in range(horizon - 2, -1, -1):
        c_t = np.zeros((n_z, n_m))
        
        for z in range(n_z):
            for i, m in enumerate(m_grid):
                # Find optimal c for state (z, m) at time t
                # m' = R * (m - c) + income[z']
                # z' drawn from transition[z]
                
                # Try a grid of consumption values
                # c in [epsilon, m]
                n_c = budget
                c_vals = m * np.linspace(0.001, 1.0, n_c)
                c_vals = np.maximum(c_vals, 1e-12)
                
                best_val = -np.inf
                best_c = c_vals[0]
                
                for c in c_vals:
                    m_save = m - c
                    if m_save < -1e-10:
                        continue
                    
                    # Expected continuation value
                    cont_val = 0.0
                    for zp in range(n_z):
                        prob = transition[z, zp]
                        if prob < 1e-10:
                            continue
                        m_next = R * m_save + income[zp]
                        
                        # Interpolate V_next at m_next
                        if m_next <= m_grid[0]:
                            v_interp = V_next[zp, 0]
                        elif m_next >= m_grid[-1]:
                            v_interp = V_next[zp, -1]
                        else:
                            v_interp = np.interp(m_next, m_grid, V_next[zp])
                        
                        cont_val += prob * v_interp
                    
                    total_val = utility(c) + beta * cont_val
                    
                    if total_val > best_val:
                        best_val = total_val
                        best_c = c
                
                c_t[z, i] = best_c
                V_next[z, i] = best_val
        
        c_policies[t] = c_t
        
        # Check time
        if time.time() > deadline:
            # Return what we have so far
            break
    
    # Now interpolate policies onto resource_grid
    result = np.zeros((horizon, n_z, len(resource_grid)))
    
    for t in range(horizon):
        if c_policies[t] is None:
            # Use final period policy
            c_t = c_final
        else:
            c_t = c_policies[t]
        
        for z in range(n_z):
            # Interpolate c_t[z, :] from m_grid to resource_grid
            result[t, z, :] = np.interp(resource_grid, m_grid, c_t[z, :])
    
    # Enforce constraints: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            for i in range(len(resource_grid)):
                m_val = resource_grid[i]
                c_val = result[t, z, i]
                if c_val <= 0:
                    result[t, z, i] = 1e-12
                if c_val > m_val + 1e-10:
                    result[t, z, i] = m_val
    
    # Final period must equal grid within 1e-9
    for z in range(n_z):
        result[horizon - 1, z, :] = resource_grid
    
    return result
# EVOLVE-BLOCK-END
