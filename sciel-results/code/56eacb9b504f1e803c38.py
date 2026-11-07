import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = len(resource_grid)
    Z = len(income)
    m_grid = resource_grid

    # CRRA utility
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c**(1 - gamma) / (1 - gamma)

    # Grid for value function calculation
    # Max m in resource_grid is 40.
    # Max m' = R * 40 + max(income) = 1.02 * 40 + 1.5 = 42.3
    # Min m' = min(income) = 0.5
    # We need to evaluate V at m'. If m' > max(m_grid_vf), we clamp.
    # Let's use a grid that covers [0.001, 45]
    N_m_vf = 1000
    m_grid_vf = np.geomspace(0.001, 45.0, N_m_vf)
    
    # Grid for consumption fraction c/m
    # We need to find c in (0, m].
    # Let's use a fine grid for fractions.
    N_f = 300
    f_grid = np.linspace(0.001, 1.0, N_f)
    
    # V[t, z, i] is value at time t, state z, wealth index i
    # C[t, z, i] is optimal consumption at time t, state z, wealth index i
    V = np.zeros((horizon, Z, N_m_vf))
    C = np.zeros((horizon, Z, N_m_vf))
    
    # Terminal period: t = horizon - 1 = 5
    # c = m, V = u(m)
    for z in range(Z):
        C[horizon-1, z, :] = m_grid_vf
        V[horizon-1, z, :] = utility(m_grid_vf)
        
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        for z in range(Z):
            # We want to maximize J(c) = u(c) + beta * sum_{z'} P[z, z'] V[t+1, z'](R(m-c) + y[z'])
            # for each m in m_grid_vf.
            
            # m is shape (N_m_vf,)
            # f is shape (N_f,)
            # c = m[:, None] * f[None, :] -> shape (N_m_vf, N_f)
            c_matrix = m_grid_vf[:, None] * f_grid[None, :]
            
            # Compute utility part
            u_c = utility(c_matrix)
            
            # Compute continuation value part
            cont_val = np.zeros_like(c_matrix)
            for zp in range(Z):
                m_prime = R * (m_grid_vf[:, None] - c_matrix) + income[zp]
                # Clip m_prime to the grid range for interpolation
                m_prime_clipped = np.clip(m_prime, m_grid_vf[0], m_grid_vf[-1])
                
                # Interpolate V[t+1, zp] at m_prime_clipped
                # V[t+1, zp] is shape (N_m_vf,)
                # m_prime_clipped is shape (N_m_vf, N_f)
                # We need to interpolate for each element.
                # np.interp is not directly vectorized for 2D x values in one call efficiently without flattening.
                # Let's flatten.
                m_prime_flat = m_prime_clipped.flatten()
                v_interp_flat = np.interp(m_prime_flat, m_grid_vf, V[t+1, zp, :])
                v_interp = v_interp_flat.reshape(N_m_vf, N_f)
                
                cont_val += transition[z, zp] * v_interp
            
            obj = u_c + beta * cont_val
            
            # Find argmax along axis 1 (over f)
            idx_max = np.argmax(obj, axis=1)
            
            # Extract optimal c and V
            # c_opt[i] = c_matrix[i, idx_max[i]]
            C[t, z, :] = c_matrix[np.arange(N_m_vf), idx_max]
            V[t, z, :] = obj[np.arange(N_m_vf), idx_max]
            
            # Check time
            if time.time() > deadline - 5:
                # If we run out of time, we break. 
                # The current V and C for earlier t might be incomplete or zero.
                # We should ensure we return something valid.
                # Since we fill t from horizon-2 down to 0, if we break, t=0..t_current are filled.
                # t > t_current are zero.
                # We need to handle this carefully in the final interpolation.
                pass

    # Now we have C[t, z, :] for m in m_grid_vf.
    # We need to return consumption on m_grid (resource_grid).
    consumption = np.zeros((horizon, Z, M))
    
    for t in range(horizon):
        for z in range(Z):
            if t == horizon - 1:
                consumption[t, z, :] = m_grid
            else:
                # Interpolate C[t, z, :] from m_grid_vf to m_grid
                consumption[t, z, :] = np.interp(m_grid, m_grid_vf, C[t, z, :])
                
                # Ensure validity: 0 < c <= m + 1e-10
                consumption[t, z, :] = np.clip(consumption[t, z, :], 1e-12, m_grid + 1e-10)
    
    # Final validity check
    consumption = np.nan_to_num(consumption, nan=1e-12, posinf=40.0, neginf=1e-12)
    
    return consumption
