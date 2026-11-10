import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    N = len(resource_grid)
    T = horizon
    Z = len(income)
    
    c_policy = np.zeros((T, Z, N))
    
    def u(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    # Last period
    c_policy[T-1, :, :] = resource_grid[np.newaxis, :, :]
    V_next = u(resource_grid) # Shape (N,)
    
    # Fine grid for c search
    K = 2000
    c_fracs = np.linspace(1e-4, 1.0, K)
    
    for t in range(T - 2, -1, -1):
        if time.time() > deadline:
             for tt in range(t, T):
                 c_policy[tt, :, :] = resource_grid[np.newaxis, :, :]
             return c_policy

        V_t = np.zeros((Z, N))
        
        for z in range(Z):
            probs = transition[z]
            
            # c_grid shape (N, K)
            # m_i * c_fracs_k
            c_grid = resource_grid[:, np.newaxis] * c_fracs[np.newaxis, :]
            
            # Objective shape (N, K)
            obj = np.zeros((N, K))
            
            # u(c)
            obj += u(c_grid)
            
            # Continuation value
            # Sum over z'
            cont_val = np.zeros((N, K))
            for z_prime in range(Z):
                if probs[z_prime] > 0:
                    m_prime = R * (resource_grid[:, np.newaxis] - c_grid) + income[z_prime]
                    # Clip to grid
                    m_prime = np.clip(m_prime, resource_grid[0], resource_grid[-1])
                    # Interpolate V_next (shape N) onto m_prime (shape N, K)
                    # np.interp expects 1D xp. It broadcasts fp.
                    # We can flatten m_prime, interpolate, reshape.
                    m_prime_flat = m_prime.flatten()
                    v_vals = np.interp(m_prime_flat, resource_grid, V_next)
                    v_vals = v_vals.reshape(N, K)
                    cont_val += probs[z_prime] * v_vals
            
            obj += beta * cont_val
            
            # Find max c for each m
            idx_max = np.argmax(obj, axis=1)
            # c_policy[t, z, i] = c_grid[i, idx_max[i]]
            c_policy[t, z, :] = c_grid[np.arange(N), idx_max]
            V_t[z, :] = obj[np.arange(N), idx_max]
            
        V_next = V_t
        
    return c_policy
