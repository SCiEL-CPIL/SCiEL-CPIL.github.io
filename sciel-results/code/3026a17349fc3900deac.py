import numpy as np

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    N_m = len(resource_grid)
    N_z = len(income)
    
    # Precompute utility function
    def util(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    # V[t, m_idx, z]
    V = np.zeros((horizon, N_m, N_z))
    # C[t, z, m_idx] - Note the axis order for return
    C = np.zeros((horizon, N_z, N_m))
    
    # Final period t = horizon - 1
    t = horizon - 1
    # Consume everything
    C[t, :, :] = resource_grid[None, :]
    V[t, :, :] = util(resource_grid)[:, None] # util returns (N_m,), need (N_m, 1) to broadcast to (N_m, N_z)
    # Actually V[t, m_idx, z] should be same for all z in last period?
    # V_5(m, z) = u(m). Yes.
    V[t, :, :] = util(resource_grid)[:, np.newaxis]
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        for z in range(N_z):
            # We need to find optimal c for each m
            # m is resource_grid
            # Candidates for c: m * linspace(1e-4, 1, 300)
            # To vectorize over m, we can create a 2D array of candidates
            # But m varies, so c_candidates shape (N_m, N_cand)
            
            # Let's loop over m to keep memory low and code simple, 
            # or vectorize if possible. 
            # N_m=101, N_cand=300. 101*300 = 30k floats. Very small.
            # Let's vectorize over m for speed.
            
            # c_cands shape (N_m, N_cand)
            # c_cands[i, j] = resource_grid[i] * cand_frac[j]
            cand_frac = np.linspace(1e-4, 1.0, 300)
            c_cands = resource_grid[:, None] * cand_frac[None, :]
            
            # m_prime shape (N_m, N_cand, N_z_prime)
            # m_prime[i, j, zp] = R * (m_i - c_cands[i, j]) + income[zp]
            m_prime = R * (resource_grid[:, None, None] - c_cands[:, :, None]) + income[None, None, :]
            
            # Continuation Value
            # V_next[m_idx, z_prime] is V[t+1, :, zp]
            # We need to interpolate V[t+1, :, zp] at m_prime[i, j, zp]
            
            # np.interp works on 1D arrays.
            # We can loop over z_prime (0 and 1) and accumulate.
            
            V_cont = np.zeros_like(c_cands) # (N_m, N_cand)
            
            for zp in range(N_z):
                # Prob of transitioning from z to zp
                prob = transition[z, zp]
                if prob == 0:
                    continue
                
                # Interpolate V[t+1, :, zp] at m_prime[:, :, zp]
                # m_prime_flat = m_prime[:, :, zp].flatten()
                # V_flat = V[t+1, :, zp]
                # V_interp = np.interp(m_prime_flat, resource_grid, V_flat)
                # V_interp = V_interp.reshape(N_m, N_cand)
                
                # Vectorized interp? np.interp doesn't support 2D x directly in all versions efficiently, 
                # but we can reshape.
                m_prime_flat = m_prime[:, :, zp].flatten()
                V_flat = V[t+1, :, zp]
                V_interp = np.interp(m_prime_flat, resource_grid, V_flat)
                V_interp = V_interp.reshape(N_m, -1)
                
                V_cont += prob * V_interp
            
            # Objective = u(c) + beta * V_cont
            u_c = util(c_cands)
            objective = u_c + beta * V_cont
            
            # Find max over candidates (axis 1)
            max_idx = np.argmax(objective, axis=1)
            
            # Get optimal c
            # c_opt[i] = c_cands[i, max_idx[i]]
            c_opt = c_cands[np.arange(N_m), max_idx]
            
            # Get optimal value
            V_opt = objective[np.arange(N_m), max_idx]
            
            # Store
            C[t, z, :] = c_opt
            V[t, :, z] = V_opt
            
    return C
