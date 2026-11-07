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
    
    # Internal grid for value function
    # Use a finer grid for better interpolation accuracy
    n_m_int = 6000
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
    
    # Grid for consumption fraction s = c/m
    # Use a finer grid for better policy resolution
    n_s = 2000
    s_grid = np.linspace(0.0001, 1.0, n_s)
    
    # Pre-compute transition probabilities weighted by beta
    beta_trans = beta * transition
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            # m_internal shape (n_m_int,)
            # We want to find optimal s for each m
            # c = m * s
            # m_next = R * (m - c) + income[z']
            
            m_col = m_internal[:, np.newaxis]  # (n_m_int, 1)
            s_row = s_grid[np.newaxis, :]      # (1, n_s)
            
            c_mat = m_col * s_row              # (n_m_int, n_s)
            c_mat = np.maximum(c_mat, 1e-12)
            
            u_c = utility(c_mat)               # (n_m_int, n_s)
            
            total_val = np.zeros_like(u_c)     # (n_m_int, n_s)
            
            for zp in range(n_z):
                # m_save = R * (m - c)
                # m_next = m_save + income[zp]
                m_next = R * (m_col - c_mat) + income[zp]  # (n_m_int, n_s)
                
                # Interpolate V[t+1, zp, :] at m_next
                # V_next is defined on m_internal
                # np.interp works on 1D. We flatten m_next.
                m_next_flat = m_next.flatten()
                # Clip m_next to the range of m_internal to avoid extrapolation issues
                # np.interp does linear extrapolation by default if not clipped? 
                # Actually np.interp uses the first/last value for out-of-range.
                # It's better to clip to [m_internal[0], m_internal[-1]]
                m_next_clipped = np.clip(m_next_flat, m_internal[0], m_internal[-1])
                
                v_next_flat = np.interp(m_next_clipped, m_internal, V[t + 1, zp, :])
                v_next = v_next_flat.reshape(m_next.shape)
                
                total_val += beta_trans[z, zp] * v_next
            
            total_val += u_c
            
            # Find best s for each m
            best_idx = np.argmax(total_val, axis=1)
            V[t, z, :] = total_val[np.arange(n_m_int), best_idx]
            C[t, z, :] = c_mat[np.arange(n_m_int), best_idx]
            
            # Enforce c <= m
            C[t, z, :] = np.minimum(C[t, z, :], m_internal)
            C[t, z, :] = np.maximum(C[t, z, :], 1e-12)

    # Interpolate policy C from m_internal to m_grid_out
    consumption = np.zeros((horizon, n_z, n_m_out))
    
    for t in range(horizon):
        for z in range(n_z):
            # Interpolate C[t, z, :] (defined on m_internal) to m_grid_out
            c_out = np.interp(m_grid_out, m_internal, C[t, z, :])
            consumption[t, z, :] = c_out
            
            # Enforce constraints on the output grid
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid_out)
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            
    # Terminal period must consume everything exactly
    consumption[horizon - 1, :, :] = m_grid_out[np.newaxis, :]
    
    # Final polish: for each (t, z), do a local refinement on the output grid
    # Use the value function to check if we can improve
    if time.time() - start_time < deadline - 5:
        # For each state, try a finer search around the interpolated policy
        for t in range(horizon - 1):  # Skip terminal period
            for z in range(n_z):
                m_out = m_grid_out
                c_current = consumption[t, z, :]
                
                # For each m, do a local search with finer s grid
                # Only do this if we have enough time
                if time.time() - start_time > deadline - 3:
                    break
                
                # Use a local refinement: for each m, search s in [s_center - delta, s_center + delta]
                # where s_center = c_current / m_out
                s_center = c_current / m_out
                s_center = np.clip(s_center, 0.0001, 1.0)
                
                # Local grid: 50 points around the center
                delta_s = 0.01
                s_local = np.linspace(0, 50)
                for i_m in range(n_m_out):
                    if time.time() - start_time > deadline - 2:
                        break
                    
                    sc = s_center[i_m]
                    s_lo = max(0.0001, sc - delta_s)
                    s_hi = min(1.0, sc + delta_s)
                    s_fine = np.linspace(s_lo, s_hi, 50)
                    
                    m_val = m_out[i_m]
                    c_fine = m_val * s_fine
                    c_fine = np.maximum(c_fine, 1e-12)
                    
                    u_fine = utility(c_fine)
                    total_fine = np.zeros_like(u_fine)
                    
                    for zp in range(n_z):
                        m_next_fine = R * (m_val - c_fine) + income[zp]
                        m_next_clipped = np.clip(m_next_fine, m_internal[0], m_internal[-1])
                        v_next_fine = np.interp(m_next_clipped, m_internal, V[t + 1, zp, :])
                        total_fine += beta_trans[z, zp] * v_next_fine
                    
                    total_fine += u_fine
                    
                    best_local_idx = np.argmax(total_fine)
                    best_c = c_fine[best_local_idx]
                    
                    # Update if better
                    if total_fine[best_local_idx] > total_val[i_m, np.argmax(total_val[i_m])]:
                        c_current[i_m] = best_c
                        c_current[i_m] = np.clip(c_current[i_m], 1e-12, m_val)
                
                consumption[t, z, :] = c_current
    
    return consumption
