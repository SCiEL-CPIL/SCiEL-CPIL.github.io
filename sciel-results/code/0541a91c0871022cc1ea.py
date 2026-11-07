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

    # Internal grid for value function - use a fine grid
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

    # Pre-compute transition probabilities weighted by beta
    beta_trans = beta * transition

    # Backward induction with adaptive s-grid refinement
    for t in range(horizon - 2, -1, -1):
        # First pass: coarse s-grid to find approximate optimal s for each m
        n_s_coarse = 200
        s_coarse = np.linspace(0.001, 1.0, n_s_coarse)
        
        # Store approximate best s for each m and z
        best_s_approx = np.zeros((n_z, n_m_int))
        
        for z in range(n_z):
            m_col = m_internal[:, np.newaxis]  # (n_m_int, 1)
            s_row = s_coarse[np.newaxis, :]    # (1, n_s_coarse)
            
            c_mat = m_col * s_row              # (n_m_int, n_s_coarse)
            c_mat = np.maximum(c_mat, 1e-12)
            
            u_c = utility(c_mat)               # (n_m_int, n_s_coarse)
            
            total_val = np.zeros_like(u_c)     # (n_m_int, n_s_coarse)
            
            for zp in range(n_z):
                m_next = R * (m_col - c_mat) + income[zp]  # (n_m_int, n_s_coarse)
                m_next_flat = m_next.flatten()
                m_next_clipped = np.clip(m_next_flat, m_internal[0], m_internal[-1])
                v_next_flat = np.interp(m_next_clipped, m_internal, V[t + 1, zp, :])
                v_next = v_next_flat.reshape(m_next.shape)
                total_val += beta_trans[z, zp] * v_next
            
            total_val += u_c
            
            best_idx = np.argmax(total_val, axis=1)
            V[t, z, :] = total_val[np.arange(n_m_int), best_idx]
            best_s_approx[z, :] = s_coarse[best_idx]
        
        # Second pass: fine s-grid centered around the approximate optimum
        n_s_fine = 300
        for z in range(n_z):
            # For each m, create a local fine grid around the approximate optimum
            # Use a window of about 20 coarse points on each side
            window = 20
            s_fine_local = []
            c_fine_local = []
            total_val_fine = np.zeros((n_m_int, n_s_fine))
            
            for i_m in range(n_m_int):
                s_center = best_s_approx[z, i_m]
                # Create a local grid around s_center
                s_lower = max(0.001, s_center - 0.05)
                s_upper = min(1.0, s_center + 0.05)
                s_local = np.linspace(s_lower, s_upper, n_s_fine)
                
                m_val = m_internal[i_m]
                c_local = np.maximum(m_val * s_local, 1e-12)
                
                u_local = utility(c_local)
                
                total_val_local = np.zeros(n_s_fine)
                
                for zp in range(n_z):
                    m_next_local = R * (m_val - c_local) + income[zp]
                    m_next_clipped = np.clip(m_next_local, m_internal[0], m_internal[-1])
                    v_next_local = np.interp(m_next_clipped, m_internal, V[t + 1, zp, :])
                    total_val_local += beta_trans[z, zp] * v_next_local
                
                total_val_local += u_local
                total_val_fine[i_m, :] = total_val_local
            
            # Find best s in the fine grid
            best_idx_fine = np.argmax(total_val_fine, axis=1)
            V[t, z, :] = total_val_fine[np.arange(n_m_int), best_idx_fine]
            
            # Get the optimal c
            for i_m in range(n_m_int):
                s_center = best_s_approx[z, i_m]
                s_lower = max(0.001, s_center - 0.05)
                s_upper = min(1.0, s_center + 0.05)
                s_local = np.linspace(s_lower, s_upper, n_s_fine)
                C[t, z, i_m] = m_internal[i_m] * s_local[best_idx_fine[i_m]]
            
            # Enforce c <= m
            C[t, z, :] = np.minimum(C[t, z, :], m_internal)
            C[t, z, :] = np.maximum(C[t, z, :], 1e-12)
        
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
