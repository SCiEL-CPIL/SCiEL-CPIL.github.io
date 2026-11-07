import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    M = resource_grid
    num_m = len(M)
    
    # Use a fine grid for internal computation
    M_fine = np.geomspace(0.001, 40, 1200)
    num_m_fine = len(M_fine)
    
    # Number of consumption candidates per wealth level
    K = 120
    
    # Fractions of wealth to consume
    fracs = np.linspace(0.001, 1.0, K)
    
    # V[t, z, i] is the value function on fine grid
    # C[t, z, i] is the consumption policy on fine grid
    V = np.zeros((horizon, 2, num_m_fine))
    C = np.zeros((horizon, 2, num_m_fine))
    
    # Final period t = horizon - 1
    V[horizon-1, :, :] = 0.0
    C[horizon-1, :, :] = M_fine
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            for tt in range(t, -1, -1):
                if tt + 1 < horizon:
                    C[tt, :, :] = C[tt+1, :, :]
                    V[tt, :, :] = V[tt+1, :, :]
                else:
                    C[tt, :, :] = M_fine
                    V[tt, :, :] = 0.0
            C_out = np.zeros((horizon, 2, num_m))
            for tt in range(horizon):
                for z in range(2):
                    C_out[tt, z, :] = np.interp(M, M_fine, C[tt, z, :])
            C_out[horizon-1, :, :] = M
            return C_out
        
        # Precompute next-period resources for each z, z'
        # M_next[z, z', i, k] = R * (M_fine[i] - C_cand[i, k]) + income[z']
        C_cand = np.outer(M_fine, fracs)  # (num_m_fine, K)
        
        # Immediate utility
        c_safe = np.maximum(C_cand, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            u = np.log(c_safe)
        else:
            u = np.power(c_safe, 1.0 - gamma) / (1.0 - gamma)
        
        # For each current state z
        for z in range(2):
            cont_val = np.zeros((num_m_fine, K))
            
            for z_prime in range(2):
                p = transition[z, z_prime]
                if p == 0:
                    continue
                
                M_next = R * (M_fine[:, None] - C_cand) + income[z_prime]
                M_next_clipped = np.clip(M_next, M_fine[0], M_fine[-1])
                
                # Use np.interp for speed and robustness
                # Interpolate V[t+1, z_prime, :] at M_next_clipped
                V_next = V[t+1, z_prime, :]
                V_interp = np.empty_like(M_next_clipped)
                for i in range(num_m_fine):
                    V_interp[i, :] = np.interp(M_next_clipped[i, :], M_fine, V_next)
                
                cont_val += p * V_interp
            
            total_val = u + beta * cont_val
            opt_k = np.argmax(total_val, axis=1)
            indices = np.arange(num_m_fine)
            C[t, z, :] = C_cand[indices, opt_k]
            V[t, z, :] = total_val[indices, opt_k]
    
    # Interpolate back to original grid
    C_out = np.zeros((horizon, 2, num_m))
    for tt in range(horizon):
        for z in range(2):
            C_out[tt, z, :] = np.interp(M, M_fine, C[tt, z, :])
    
    C_out[horizon-1, :, :] = M
    C_out = np.clip(C_out, 1e-12, M[None, None, :] + 1e-10)
    C_out[horizon-1, :, :] = M
    
    return C_out
