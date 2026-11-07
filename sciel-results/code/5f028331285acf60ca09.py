import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']

    m_grid_out = np.array(resource_grid)
    n_m_out = len(m_grid_out)
    n_z = len(income)

    # Internal grid for value function - fine grid
    n_m_int = 8000
    m_internal = np.geomspace(0.001, 50, n_m_int)

    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    # Value function storage: V[t, z, m_idx]
    V = np.zeros((horizon, n_z, n_m_int))
    # Policy storage: C[t, z, m_idx]
    C = np.zeros((horizon, n_z, n_m_int))

    # Terminal period: consume everything
    V[horizon - 1, :, :] = utility(m_internal)[np.newaxis, :]
    C[horizon - 1, :, :] = m_internal[np.newaxis, :]

    # Pre-compute transition probabilities weighted by beta
    beta_trans = beta * transition

    # Precompute m_next for each (z, zp, m_idx) to avoid recomputation
    # m_next[z, zp, i_m] = R * (m_internal[i_m] - c) + income[zp]
    # But c depends on the optimal policy, so we can't precompute fully.
    # Instead, we'll use vectorized operations where possible.

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            # For each m, we need to find optimal s in [0.001, 1.0]
            # Use a fine s-grid and vectorize over m
            n_s = 200
            s_grid = np.linspace(0.001, 1.0, n_s)
            
            # c_grid[i_m, i_s] = m_internal[i_m] * s_grid[i_s]
            c_grid = m_internal[:, np.newaxis] * s_grid[np.newaxis, :]
            c_grid = np.maximum(c_grid, 1e-12)
            
            # u_grid[i_m, i_s] = utility(c_grid[i_m, i_s])
            u_grid = utility(c_grid)
            
            # For each s, compute continuation value
            # m_next[i_m, i_s, zp] = R * (m_internal[i_m] - c_grid[i_m, i_s]) + income[zp]
            m_next = R * (m_internal[:, np.newaxis] - c_grid) + income[np.newaxis, np.newaxis, :]
            m_next = np.clip(m_next, m_internal[0], m_internal[-1])
            
            # Interpolate V[t+1, zp, :] onto m_next
            # v_next[i_m, i_s, zp] = V[t+1, zp, :][m_next[i_m, i_s, zp]]
            v_next = np.zeros_like(m_next)
            for zp in range(n_z):
                v_next[:, :, zp] = np.interp(m_next[:, :, zp], m_internal, V[t + 1, zp, :])
            
            # Weighted sum over zp
            cont_val = np.tensordot(v_next, beta_trans[z, :], axes=([2], [0]))
            # cont_val[i_m, i_s] = sum_zp beta_trans[z, zp] * v_next[i_m, i_s, zp]
            
            # Total value for each (m, s)
            total_val = u_grid + cont_val
            
            # Find best s for each m
            best_s_idx = np.argmax(total_val, axis=1)
            best_s = s_grid[best_s_idx]
            best_c = m_internal * best_s
            best_c = np.clip(best_c, 1e-12, m_internal)
            
            C[t, z, :] = best_c
            V[t, z, :] = total_val[np.arange(n_m_int), best_s_idx]
        
        # Time check
        if time.time() - start_time > deadline - 5:
            break

    # Interpolate policy C from m_internal to m_grid_out
    consumption = np.zeros((horizon, n_z, n_m_out))

    for t in range(horizon):
        for z in range(n_z):
            c_out = np.interp(m_grid_out, m_internal, C[t, z, :])
            consumption[t, z, :] = c_out

            # Enforce constraints on the output grid
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid_out)
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)

    # Terminal period must consume everything exactly
    consumption[horizon - 1, :, :] = m_grid_out[np.newaxis, :]

    return consumption
