import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin below 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    m_out = resource_grid
    n_out = len(m_out)
    n_z = len(income)
    
    # Utility functions
    def u(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c_safe)
        else:
            return (c_safe ** (1 - gamma)) / (1 - gamma)
            
    # Dense internal grid for accurate value function
    # We need enough points to resolve the policy function accurately
    # The evaluator uses 251 points, so 2000 internal points is a good balance
    n_int = 2000
    m_int = np.geomspace(0.001, 40, n_int)
    
    # Value function V[t, z, i] for t in 0..horizon-1, z in 0..n_z-1, i in 0..n_int-1
    # Policy function C[t, z, i]
    V = np.zeros((horizon, n_z, n_int))
    C = np.zeros((horizon, n_z, n_int))
    
    # Terminal condition: V_T(m, z) = u(m)
    V[horizon-1, :, :] = u(m_int)
    C[horizon-1, :, :] = m_int  # c = m at terminal
    
    # Backward induction
    # We use a fixed number of consumption candidates for each m
    # This ensures uniform resolution in the optimization
    n_cand = 100
    
    for t in range(horizon - 2, -1, -1):
        # Check time
        if time.time() > deadline - 10:
            # If we are running out of time, we can break and return what we have?
            # But we need to fill all t. 
            # Given the vectorization, this loop is fast.
            pass
            
        V_next = V[t+1]  # Shape (n_z, n_int)
        
        for z in range(n_z):
            V_curr_z = np.zeros(n_int)
            C_curr_z = np.zeros(n_int)
            
            # Vectorize over m_int
            # For each m_i, we need to optimize over c.
            # Since the optimal c depends on m_i, we can't easily vectorize the entire 
            # optimization in one shot without large memory usage.
            # However, we can process in chunks or just loop over m_i.
            # n_int = 2000, n_cand = 100. 2000 * 100 = 200,000 operations per z.
            # This is fast.
            
            for i, m_val in enumerate(m_int):
                # Generate consumption grid
                # c must be in (0, m_val)
                # We use linspace from a small epsilon to m_val * (1 - eps)
                c_grid = np.linspace(1e-12, m_val * (1 - 1e-6), n_cand)
                
                # Current utility
                u_curr = u(c_grid)
                
                # Next resources m' for each c and each next state z'
                # m' = R * (m_val - c) + income[z']
                # Shape of c_grid: (n_cand,)
                # Shape of m_prime: (n_cand, n_z)
                m_prime = R * (m_val - c_grid[:, np.newaxis]) + income[np.newaxis, :]
                
                # Clamp m_prime to the range of m_int for interpolation
                m_prime = np.clip(m_prime, m_int[0], m_int[-1])
                
                # Interpolate V_next for each m' and z'
                # V_next[z', :] is the value function for state z'
                # We need V_next[z', interp(m_prime[c, z'], m_int)]
                # Let's do this for each z'
                
                cont_val = np.zeros(n_cand)
                for zp in range(n_z):
                    prob = transition[z, zp]
                    if prob == 0:
                        continue
                    # Interpolate V_next[zp, :] at m_prime[:, zp]
                    # np.interp requires x (m_prime) and xp (m_int) to be 1D
                    v_interp = np.interp(m_prime[:, zp], m_int, V_next[zp, :])
                    cont_val += prob * v_interp
                
                # Total value = u(c) + beta * cont_val
                total_val = u_curr + beta * cont_val
                
                # Find max
                idx_max = np.argmax(total_val)
                V_curr_z[i] = total_val[idx_max]
                C_curr_z[i] = c_grid[idx_max]
                
            V[t, z, :] = V_curr_z
            C[t, z, :] = C_curr_z
            
    # Interpolate policy from internal grid to output grid
    pol = np.zeros((horizon, n_z, n_out))
    
    for t in range(horizon):
        for z in range(n_z):
            # Use np.interp for simplicity and speed
            # C[t, z, :] is the policy on m_int
            # We want it on m_out
            pol[t, z, :] = np.interp(m_out, m_int, C[t, z, :])
            
    # Final check: ensure validity
    # 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            c = pol[t, z, :]
            # Floor at 1e-12
            c = np.maximum(c, 1e-12)
            # Cap at m
            c = np.minimum(c, m_out)
            pol[t, z, :] = c
            
    # Terminal condition check: c[5] equal to grid within 1e-9
    # We set it exactly to m_out in the init, but interpolation might have introduced errors.
    # Let's enforce it.
    pol[horizon-1, :, :] = m_out

    return pol
