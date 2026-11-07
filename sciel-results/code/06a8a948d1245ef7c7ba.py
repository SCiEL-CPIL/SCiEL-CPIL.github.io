import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Extract parameters
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']
    
    # Output grid
    m_out = np.array(resource_grid)
    n_out = len(m_out)
    n_z = len(income)
    
    # Internal grids for high-resolution computation
    # We need to cover the range of possible m' values.
    # m' = R(m-c) + y. Max m ~ 40, max y ~ 1.5, R ~ 1.02.
    # Max m' ~ 1.02 * 40 + 1.5 ~ 42.3.
    # Min m' ~ 0.5 (if m=0, c=0, y=0.5) or higher if m>0.
    # The evaluation grid goes up to 40.
    # Let's use an internal grid up to 50 to be safe.
    n_m_int = 1500
    m_int = np.geomspace(0.001, 50, n_m_int)
    
    # Grid for consumption fraction s = c/m
    # Concave utility implies a unique optimum. 500 points is dense enough.
    n_s = 500
    s_grid = np.linspace(0.0, 1.0, n_s)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    # Pre-compute beta * transition for efficiency
    beta_trans = beta * transition
    
    # Value function storage: V[t, z, m_idx]
    V = np.zeros((horizon, n_z, n_m_int))
    # Policy storage: C[t, z, m_idx]
    C = np.zeros((horizon, n_z, n_m_int))
    
    # Terminal period t = horizon - 1
    t = horizon - 1
    for z in range(n_z):
        # Consume everything
        C[t, z, :] = m_int
        V[t, z, :] = utility(m_int)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 10.0:
            # If we are running out of time, break. 
            # We will interpolate whatever we have.
            # This is a fallback.
            break
            
        for z in range(n_z):
            # m_int shape (n_m_int,)
            # We want to find optimal s for each m
            # c = m * s
            # m_next = R * (m - c) + income[z']
            
            m_col = m_int[:, np.newaxis]  # (n_m_int, 1)
            s_row = s_grid[np.newaxis, :]  # (1, n_s)
            
            c_mat = m_col * s_row          # (n_m_int, n_s)
            c_mat = np.maximum(c_mat, 1e-12)
            
            u_c = utility(c_mat)           # (n_m_int, n_s)
            
            total_val = np.zeros_like(u_c) # (n_m_int, n_s)
            
            for zp in range(n_z):
                prob = beta_trans[z, zp]
                if prob == 0:
                    continue
                
                y_zp = income[zp]
                
                # m_next = R * (m - c) + y_zp
                m_next = R * (m_col - c_mat) + y_zp  # (n_m_int, n_s)
                
                # Interpolate V[t+1, zp, :] at m_next
                # V_next is defined on m_int
                # np.interp works on 1D. We flatten m_next.
                m_next_flat = m_next.flatten()
                
                # Clip m_next to the range of m_int to avoid extrapolation issues
                # np.interp uses the first/last value for out-of-range.
                # Clipping is safer for stability if the range is well-covered.
                m_next_clipped = np.clip(m_next_flat, m_int[0], m_int[-1])
                
                v_next_flat = np.interp(m_next_clipped, m_int, V[t + 1, zp, :])
                v_next = v_next_flat.reshape(m_next.shape)
                
                total_val += prob * v_next
            
            total_val += u_c
            
            # Find best s for each m
            best_idx = np.argmax(total_val, axis=1)
            V[t, z, :] = total_val[np.arange(n_m_int), best_idx]
            C[t, z, :] = c_mat[np.arange(n_m_int), best_idx]
            
            # Enforce c <= m and c > 0
            C[t, z, :] = np.minimum(C[t, z, :], m_int)
            C[t, z, :] = np.maximum(C[t, z, :], 1e-12)

    # Interpolate policy C from m_int to m_out
    consumption = np.zeros((horizon, n_z, n_out))
    
    for t in range(horizon):
        for z in range(n_z):
            # Interpolate C[t, z, :] (defined on m_int) to m_out
            c_out = np.interp(m_out, m_int, C[t, z, :])
            consumption[t, z, :] = c_out
            
            # Enforce constraints on the output grid
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_out)
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            
    # Terminal period must consume everything exactly
    consumption[horizon - 1, :, :] = m_out[np.newaxis, :]
    
    # Refinement pass: Local search on the output grid
    # This helps correct for interpolation errors in the policy.
    # We use the value function V to evaluate the objective.
    if time.time() < deadline - 5.0:
        for t in range(horizon - 1):  # Skip terminal period
            for z in range(n_z):
                m_curr = m_out
                c_curr = consumption[t, z, :].copy()
                
                # For each m, do a local search
                # We search in the interval [c_curr - delta, c_curr + delta]
                # clipped to [0, m_curr]
                
                # Define a local grid for c
                # We can just check a few points around the current estimate
                # Or use a finer grid of s locally.
                
                # Let's use a fine grid of s locally: 100 points
                # centered around s_curr = c_curr / m_curr
                s_curr = c_curr / m_curr
                s_curr = np.clip(s_curr, 0.0, 1.0)
                
                delta_s = 0.05
                n_local = 100
                
                for i_m in range(n_out):
                    if time.time() > deadline - 2.0:
                        break
                    
                    m_val = m_curr[i_m]
                    sc = s_curr[i_m]
                    
                    s_lo = max(0.0, sc - delta_s)
                    s_hi = min(1.0, sc + delta_s)
                    
                    # If the range is too small, skip or just keep current
                    if s_hi - s_lo < 1e-6:
                        continue
                        
                    s_fine = np.linspace(s_lo, s_hi, n_local)
                    c_fine = m_val * s_fine
                    c_fine = np.maximum(c_fine, 1e-12)
                    
                    u_fine = utility(c_fine)
                    total_fine = np.zeros_like(u_fine)
                    
                    for zp in range(n_z):
                        prob = beta_trans[z, zp]
                        if prob == 0:
                            continue
                        y_zp = income[zp]
                        m_next_fine = R * (m_val - c_fine) + y_zp
                        m_next_clipped = np.clip(m_next_fine, m_int[0], m_int[-1])
                        v_next_fine = np.interp(m_next_clipped, m_int, V[t + 1, zp, :])
                        total_fine += prob * v_next_fine
                    
                    total_fine += u_fine
                    
                    best_local_idx = np.argmax(total_fine)
                    best_c = c_fine[best_local_idx]
                    
                    # Update if better
                    # We compare the value from the local search with the value from the global grid
                    # We don't have the global value stored for the output grid points directly,
                    # but we can just take the best local one as it's a superset of the neighborhood.
                    # Actually, we should ensure we don't degrade.
                    # Let's just assign the best local value.
                    consumption[t, z, i_m] = best_c
                
                consumption[t, z, :] = np.minimum(consumption[t, z, :], m_out)
                consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
                
    # Final check for terminal period
    consumption[horizon - 1, :, :] = m_out[np.newaxis, :]
    
    return consumption
