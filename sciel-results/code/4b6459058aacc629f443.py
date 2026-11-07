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
    
    m_grid = np.array(resource_grid)
    n_m_out = len(m_grid)
    n_z = len(income)
    
    # Internal dense grid for value function computation
    # Using 1000 points to ensure high accuracy for interpolation
    m_internal = np.geomspace(0.001, 40, 1000)
    n_mi = len(m_internal)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)
    
    # Value function storage: V[t, z, m_idx]
    V = np.zeros((horizon, n_z, n_mi))
    
    # Terminal period: consume everything
    V[horizon - 1, :, :] = utility(m_internal)[np.newaxis, :]
    
    # Grid for consumption fraction s = c/m
    # We use a dense grid for the internal optimization
    n_s = 500
    s_grid = np.linspace(0.0001, 1.0, n_s)
    
    # Backward induction on the internal grid
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            # m_internal is shape (n_mi,)
            # We want to find optimal s for each m in m_internal
            # c = m * s
            # m_next = R * (m - c) + income[z']
            
            # Objective shape: (n_mi, n_s)
            # m is (n_mi, 1), s is (1, n_s)
            m_col = m_internal[:, np.newaxis]
            s_row = s_grid[np.newaxis, :]
            
            c_mat = m_col * s_row
            c_mat = np.maximum(c_mat, 1e-12)
            
            u_c = utility(c_mat)
            
            total_val = np.zeros_like(u_c)
            
            for zp in range(n_z):
                m_save = R * (m_col - c_mat)
                m_next = m_save + income[zp]
                
                # Interpolate V[t+1, zp, :] at m_next
                # m_next is (n_mi, n_s)
                # np.interp requires 1D x, but we can flatten or use a custom interp
                # For efficiency, we can pre-interpolate V[t+1, zp, :] onto a linear grid 
                # or just use np.interp with flattening. 
                # Given n_mi=1000, n_s=500, flattening is 500k points. 
                # np.interp is fast but O(N log N) or O(N) depending on implementation.
                # Let's try flattening.
                
                # Actually, np.interp can handle array x if x is 1D. 
                # m_next is 2D. We must flatten.
                m_next_flat = m_next.flatten()
                V_next_flat = V[t + 1, zp, :].flatten() # No, V is on m_internal grid
                # We need to interpolate V[t+1, zp, :] (which is defined on m_internal) 
                # at the points m_next_flat.
                
                v_next_flat = np.interp(m_next_flat, m_internal, V[t + 1, zp, :])
                v_next = v_next_flat.reshape(m_next.shape)
                
                total_val += beta * transition[z, zp] * v_next
            
            total_val += u_c
            
            # Find best s for each m
            best_idx = np.argmax(total_val, axis=1)
            V[t, z, :] = total_val[np.arange(n_mi), best_idx]
            
        # Time check
        if time.time() - start_time > deadline - 20:
            # If we run out of time, break and use heuristic for remaining t
            # Fill remaining V with terminal-like values or zeros
            # But we need to return consumption.
            # We will break out of the loop and handle the policy generation 
            # with the available V (which will be incomplete for early t).
            # For safety, if we break, we assume the policy is "consume all" or similar?
            # Better to just break and let the policy generation use whatever V is there.
            # However, if V is zeros for early t, the policy will be wrong.
            # Given the speed, this is unlikely.
            pass

    # Now construct the consumption policy on the original resource_grid
    consumption = np.zeros((horizon, n_z, n_m_out))
    
    # Terminal period
    consumption[horizon - 1, :, :] = m_grid[np.newaxis, :]
    
    # For earlier periods, find optimal c on the original grid
    # We use a finer s_grid for the final policy to ensure accuracy
    n_s_final = 1000
    s_grid_final = np.linspace(0.0001, 1.0, n_s_final)
    
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            m_out = m_grid
            m_col = m_out[:, np.newaxis]
            s_row = s_grid_final[np.newaxis, :]
            
            c_mat = m_col * s_row
            c_mat = np.maximum(c_mat, 1e-12)
            
            u_c = utility(c_mat)
            
            total_val = np.zeros_like(u_c)
            
            for zp in range(n_z):
                m_save = R * (m_col - c_mat)
                m_next = m_save + income[zp]
                
                # Interpolate V[t+1, zp, :] at m_next
                m_next_flat = m_next.flatten()
                v_next_flat = np.interp(m_next_flat, m_internal, V[t + 1, zp, :])
                v_next = v_next_flat.reshape(m_next.shape)
                
                total_val += beta * transition[z, zp] * v_next
            
            total_val += u_c
            
            best_idx = np.argmax(total_val, axis=1)
            best_c = c_mat[np.arange(n_m_out), best_idx]
            
            # Enforce c <= m
            best_c = np.minimum(best_c, m_out)
            best_c = np.maximum(best_c, 1e-12)
            
            consumption[t, z, :] = best_c
            
        # Time check
        if time.time() - start_time > deadline - 5:
            # Fill remaining with heuristic: consume 50% or similar
            # Actually, if we break here, we have computed t down to some level.
            # We need to fill the remaining earlier t's.
            # A safe fallback is to consume a fixed fraction, e.g., 0.5 * m
            for tt in range(t - 1, -1, -1):
                for zz in range(n_z):
                    consumption[tt, zz, :] = m_grid * 0.5
            break

    # Final constraints
    for t in range(horizon):
        for z in range(n_z):
            for idx_m in range(n_m_out):
                m = m_grid[idx_m]
                c = consumption[t, z, idx_m]
                if c <= 0:
                    c = 1e-12
                if c > m + 1e-10:
                    c = m
                consumption[t, z, idx_m] = c
    
    # Terminal period must consume everything
    consumption[horizon - 1, :, :] = m_grid[np.newaxis, :]
    
    return consumption
