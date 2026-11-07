"""Rewritten solution for consumption-saving policy accuracy."""
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

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

    # Value function V[t, z, m] where m is index into m_grid
    # Consumption policy C[t, z, m]
    # We'll use backward induction with a fine grid for c

    # For efficiency, we'll use a logarithmic or geometric grid for c within each m
    # But the scoring uses c = m * linspace(0.001, 1, 301) for best
    # Our policy is interpolated onto geomspace(0.001, 40, 251)
    
    # Let's use the resource_grid as the m-grid for value function
    # For each (t, z, m), we need to find optimal c
    
    # Budget: number of c-points to try per (t,z,m)
    # We'll use a variable budget approach: use more points where it matters
    
    # Value function: V[t, z, m_idx]
    V = np.zeros((horizon + 1, N_z, N_m))
    C = np.zeros((horizon + 1, N_z, N_m))
    
    # Terminal condition: c[5] = m (consume everything at end)
    # Actually horizon=6, so t=0..5, terminal at t=6
    # c[5] equal to the grid within 1e-9 means at t=5, c=m
    V[5, :, :] = util(m_grid)  # Wait, at t=5, if horizon is 6, then t=5 is last period
    # Actually let me re-read: "c[5] equal to the grid within 1e-9"
    # This means at t=5 (last period), consumption equals resources
    C[5, :, :] = m_grid
    
    # Backward induction from t=4 down to t=0
    # For each t from 4 down to 0:
    #   For each z:
    #     For each m_idx:
    #       m = m_grid[m_idx]
    #       Try c in some grid
    #       For each c, next_m = R*(m-c) + income[z']
    #       z' drawn from transition[z]
    #       Value = util(c) + beta * sum_z' transition[z,z'] * V[t+1, z', next_m]
    #       V[t,z,m] = max over c
    #       C[t,z,m] = argmax c

    # To speed up, we can vectorize over m_grid for each (t,z)
    # Let's use a c-grid that's a fraction of m
    
    # The scoring interpolates our policy onto geomspace(0.001, 40, 251)
    # and compares to best over c = m * linspace(0.001, 1, 301)
    # So we want our policy to be close to the true optimum
    
    # Let's use a relatively fine c-grid. Budget is 48.
    # We'll use budget c-points per (t,z) evaluation, but vectorized over m.
    
    # Actually, let's think about this differently.
    # For each (t, z), we have a function V[t+1, z', m'] for each z'
    # We need to compute V[t,z,m] = max_c { util(c) + beta * sum_z' P[z,z'] * V[t+1,z', R*(m-c)+income[z']] }
    
    # Let's precompute the continuation value as a function of m' for each z'
    # Then for each m, we evaluate over c
    
    # Use a c-grid: c = m * linspace(0.001, 1, budget)
    # But we need to be careful about the interpolation of V[t+1]
    
    # Let's use np.interp for the continuation value
    # V[t+1, z', :] is defined on m_grid
    # We need to evaluate it at next_m values
    
    # For efficiency, let's batch the c evaluations
    
    for t in range(4, -1, -1):
        for z in range(N_z):
            # For each m in m_grid, find optimal c
            # Vectorize over m_grid
            # c_frac = linspace(0.001, 1, budget)
            c_frac = np.linspace(0.001, 1.0, budget)
            # c = m_grid[:, None] * c_frac[None, :]  shape (N_m, budget)
            c_vals = m_grid[:, None] * c_frac[None, :]
            
            # next_m = R * (m - c) + income[z']
            # next_m shape: (N_m, budget, N_z) for each z'
            next_m = R * (m_grid[:, None, None] - c_vals[:, :, None]) + income[None, None, :]
            
            # Clip next_m to be >= 0 (should be since c <= m)
            next_m = np.maximum(next_m, 0.0)
            
            # Evaluate V[t+1, z', next_m] for each z'
            # V[t+1, z', :] is on m_grid
            # We need to interpolate
            # next_m shape: (N_m, budget, N_z)
            # For each z', interpolate V[t+1, z', :] at next_m[:,:,z']
            
            # Vectorize interpolation
            # np.interp doesn't vectorize easily, so let's do it per z'
            cont_val = np.zeros((N_m, budget))
            for zp in range(N_z):
                # next_m[:, :, zp] shape (N_m, budget)
                # V[t+1, zp, :] shape (N_m,)
                # Interpolate each row
                # np.interp works on 1D, so we need to apply per row
                # Let's use a trick: flatten and use np.interp, then reshape
                nm_flat = next_m[:, :, zp].ravel()
                v_flat = np.interp(nm_flat, m_grid, V[t+1, zp, :])
                cont_val[:, :, ] += transition[z, zp] * v_flat.reshape(N_m, budget)
            
            # Actually, let me redo this more carefully
            cont_val = np.zeros((N_m, budget))
            for zp in range(N_z):
                nm = next_m[:, :, zp]  # (N_m, budget)
                # Interpolate V[t+1, zp, :] at nm
                # For each m_idx, interpolate at nm[m_idx, :]
                v_interp = np.zeros((N_m, budget))
                for mi in range(N_m):
                    v_interp[mi, :] = np.interp(nm[mi, :], m_grid, V[t+1, zp, :])
                cont_val += transition[z, zp] * v_interp
            
            # Total value: util(c) + beta * cont_val
            u_c = util(c_vals)
            total_val = u_c + beta * cont_val
            
            # Find max over c for each m
            best_idx = np.argmax(total_val, axis=1)
            V[t, z, :] = total_val[np.arange(N_m), best_idx]
            C[t, z, :] = c_vals[np.arange(N_m), best_idx]
        
        # Check deadline
        if time.time() > DEADLINE - 5:
            break
    
    # Ensure c[5] = m_grid
    C[5, :, :] = m_grid
    V[5, :, :] = util(m_grid)
    
    # Ensure finite values and 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(N_z):
            C[t, z, :] = np.clip(C[t, z, :], 1e-12, m_grid + 1e-10)
            C[t, z, :] = np.where(np.isfinite(C[t, z, :]), C[t, z, :], m_grid)
    
    # Return C[0:6, :, :] shape (6, 2, 101)
    return C[0:6, :, :]
