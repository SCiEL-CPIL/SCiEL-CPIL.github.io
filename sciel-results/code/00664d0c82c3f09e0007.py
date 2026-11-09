"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Grid for search
    m_grid = resource_grid
    n_m = len(m_grid)
    n_z = len(income)
    
    # Value function storage: V[t, z, m]
    V = np.zeros((horizon + 1, n_z, n_m))
    
    # Consumption policy: C[t, z, m]
    C = np.zeros((horizon, n_z, n_m))
    
    # Final period: consume everything
    for z in range(n_z):
        C[horizon - 1, z, :] = m_grid
        V[horizon, z, :] = _utility(m_grid, gamma)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        # Check time
        if time.time() > deadline:
            # Return what we have so far (should be valid if we got this far)
            return _finalize(C, m_grid, horizon, n_z, n_m)
        
        for z in range(n_z):
            # For each m in the grid, find optimal c
            for i_m in range(n_m):
                m = m_grid[i_m]
                if m <= 0:
                    C[t, z, i_m] = 0.0
                    continue
                
                # Search over c in [epsilon, m]
                # Use a fine grid for c
                c_grid = np.geomspace(1e-6, m, 100)
                c_grid = np.clip(c_grid, 1e-12, m)
                
                best_c = c_grid[0]
                best_val = -np.inf
                
                for c in c_grid:
                    m_next = R * (m - c)
                    # Expected continuation value
                    cont_val = 0.0
                    for z_next in range(n_z):
                        prob = transition[z, z_next]
                        m_next_z = m_next + income[z_next]
                        # Interpolate V[t+1, z_next, m_next_z]
                        v_next = np.interp(m_next_z, m_grid, V[t + 1, z_next, :])
                        cont_val += prob * v_next
                    total_val = _utility(c, gamma) + beta * cont_val
                    if total_val > best_val:
                        best_val = total_val
                        best_c = c
                
                C[t, z, i_m] = best_c
                V[t, z, i_m] = best_val
    
    return _finalize(C, m_grid, horizon, n_z, n_m)

def _utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if np.isclose(gamma, 1.0):
        return np.log(c)
    else:
        return c ** (1 - gamma) / (1 - gamma)

def _finalize(C, m_grid, horizon, n_z, n_m):
    # Ensure c[5] (final period) equals grid
    C[horizon - 1, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    # Ensure 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            m_vals = m_grid
            c_vals = C[t, z, :]
            # Clip to valid range
            c_vals = np.clip(c_vals, 1e-12, m_vals + 1e-10)
            # Where m is very small, c should be small
            C[t, z, :] = c_vals
    
    return C
# EVOLVE-BLOCK-END
