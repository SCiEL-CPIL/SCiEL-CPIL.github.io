import numpy as np

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_m = len(resource_grid)
    n_z = 2
    
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
    
    V = np.zeros((horizon, n_z, n_m))
    c_opt = np.zeros((horizon, n_z, n_m))
    
    # Last period
    for z in range(n_z):
        c_opt[horizon - 1, z, :] = resource_grid
        V[horizon - 1, z, :] = util(resource_grid)
    
    # Number of consumption candidates
    n_cand = 2000
    # Fractions of m to consume. 
    # We need to handle the case where c > m, but by using fractions 0..1, c <= m.
    # We also need c >= 1e-12.
    # For very small m, m * 1.0 might be < 1e-12? No, resource_grid starts at 0.001.
    # 0.001 * 1.0 = 0.001 > 1e-12. So c is always >= 0.
    # We should ensure the lower bound of c is respected. 
    # The grid fractions should probably start at a small epsilon or 0.
    # If fraction is 0, c=0, util(0) is -inf (or very negative). 
    # Let's use fractions from 0 to 1.
    fractions = np.linspace(0.0, 1.0, n_cand)
    
    M = resource_grid # Shape (n_m,)
    
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            # C shape: (n_cand, n_m)
            # C[i, j] = fractions[i] * M[j]
            C = fractions[:, np.newaxis] * M[np.newaxis, :]
            
            # Compute M' for each z'
            # M' shape: (n_z, n_cand, n_m)
            # M'[zp, i, j] = R * (M[j] - C[i, j]) + income[zp]
            
            # We can compute this as:
            # M_minus_C = M[np.newaxis, np.newaxis, :] - C[np.newaxis, :, :]
            # This might be memory heavy: 2 * 2000 * 101 ~ 400k floats. Very small.
            
            M_prime = np.zeros((n_z, n_cand, n_m))
            for zp in range(n_z):
                # M[j] - C[i, j]
                # M is (n_m,), C is (n_cand, n_m)
                # M[np.newaxis, :] - C -> (n_cand, n_m)
                remaining = M[np.newaxis, :] - C
                M_prime[zp, :, :] = R * remaining + income[zp]
            
            # Interpolate V[t+1, zp, :] at M_prime[zp, :, :]
            # V_next shape: (n_z, n_m)
            V_next = V[t+1] # (n_z, n_m)
            
            # We need to clip M_prime to the grid range for interpolation
            m_min = resource_grid[0]
            m_max = resource_grid[-1]
            
            V_interp = np.zeros((n_z, n_cand, n_m))
            for zp in range(n_z):
                # Clip M_prime[zp] to [m_min, m_max]
                mp_clipped = np.clip(M_prime[zp], m_min, m_max)
                # np.interp requires 1D x, but we have (n_cand, n_m).
                # We can flatten, interp, and reshape.
                flat_mp = mp_clipped.flatten()
                flat_v = V_next[zp].flatten() # Wait, V_next[zp] is (n_m,), so flat is same.
                # Actually, np.interp(x, xp, fp) works on 1D arrays.
                # We can use np.interp on the flattened array.
                interp_flat = np.interp(flat_mp, resource_grid, V_next[zp])
                V_interp[zp, :, :] = interp_flat.reshape(n_cand, n_m)
            
            # Compute expected continuation value
            # cont shape: (n_cand, n_m)
            # cont[i, j] = sum_z' transition[z, z'] * V_interp[z', i, j]
            # V_interp is (n_z, n_cand, n_m)
            # We want to sum over axis 0 (z'), weighted by transition[z, :]
            
            # transition[z] shape: (n_z,)
            # We can do: np.tensordot(transition[z], V_interp, axes=([0], [0]))
            # This results in shape (n_cand, n_m)
            cont = np.tensordot(transition[z], V_interp, axes=([0], [0]))
            
            # Compute Objective
            # Obj = util(C) + beta * cont
            # util(C) shape: (n_cand, n_m)
            obj = util(C) + beta * cont
            
            # Find best index for each m (column)
            best_idx = np.argmax(obj, axis=0) # Shape (n_m,)
            
            # Extract optimal c
            # c_opt[t, z, j] = C[best_idx[j], j]
            c_opt[t, z, :] = C[best_idx, np.arange(n_m)]
            V[t, z, :] = obj[best_idx, np.arange(n_m)]
            
    return c_opt
