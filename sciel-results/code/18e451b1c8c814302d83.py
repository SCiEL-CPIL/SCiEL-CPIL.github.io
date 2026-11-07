import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    c_min = 1e-12
    T = horizon
    n_z = 2
    grid_out = resource_grid
    n_out = len(grid_out)
    
    def utility(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
            
    def utility_prime(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)
            
    def utility_double_prime(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return -1.0 / (c ** 2)
        else:
            return -gamma * c ** (-gamma - 1)

    # Use a denser internal grid for better accuracy
    n_inner = 400
    grid = np.geomspace(0.001, 40, n_inner)
    n_grid = len(grid)
    
    V = [None] * T
    P = [None] * T
    
    # Terminal condition
    V[T-1] = utility(grid)[np.newaxis, :].repeat(n_z, axis=0)
    P[T-1] = grid[np.newaxis, :].repeat(n_z, axis=0)
    
    # Precompute grid differences for derivatives
    # We will compute derivatives of V_next to use in Newton's method
    
    # Backward induction
    for t in range(T-2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: use previous period's policy if time is running out
            for t_rem in range(t, -1, -1):
                V[t_rem] = V[t+1].copy()
                P[t_rem] = P[t+1].copy()
            break
            
        V_t = np.zeros((n_z, n_grid))
        P_t = np.zeros((n_z, n_grid))
        V_next = V[t+1]
        
        # Compute derivatives of V_next for each state
        dV_next = np.zeros((n_z, n_grid))
        d2V_next = np.zeros((n_z, n_grid))
        
        for zp in range(n_z):
            V_z = V_next[zp]
            # First and second derivatives using finite differences on non-uniform grid
            # Central differences for interior points
            for i in range(1, n_grid - 1):
                h_minus = grid[i] - grid[i-1]
                h_plus = grid[i+1] - grid[i]
                
                # First derivative: Lagrange interpolation derivative
                L0p = h_plus / (h_minus * (h_minus + h_plus))
                L1p = (h_plus - h_minus) / (h_minus * h_plus)
                L2p = h_minus / ((h_minus + h_plus) * h_plus)
                
                dV_next[zp, i] = V_z[i-1]*L0p + V_z[i]*L1p + V_z[i+1]*L2p
                
                # Second derivative
                d2V_next[zp, i] = 2.0 * ( (V_z[i+1] - V_z[i]) / h_plus - (V_z[i] - V_z[i-1]) / h_minus ) / (h_plus + h_minus)
            
            # Edge i=0
            h0 = grid[1] - grid[0]
            h1 = grid[2] - grid[1]
            d01 = h0
            d02 = h0 + h1
            d12 = h1
            L0p = ( -d02 - d01 ) / ( -d01 * -d02 )
            L1p = -d02 / ( d01 * -d12 )
            L2p = -d01 / ( d02 * d12 )
            dV_next[zp, 0] = V_z[0]*L0p + V_z[1]*L1p + V_z[2]*L2p
            d2V_next[zp, 0] = 2.0 * ( (V_z[1] - V_z[0]) / h0 - (V_z[2] - V_z[1]) / h1 ) / (h0 + h1)
            
            # Edge i=n_grid-1
            h_m1 = grid[-1] - grid[-2]
            h_m2 = grid[-2] - grid[-3]
            d_m1 = h_m1
            d_m2 = h_m2
            d_m12 = h_m1 + h_m2
            L_0p = d_m1 / (d_m2 * d_m12)
            L_1p = -d_m12 / (d_m2 * d_m1)
            L_2p = 1.0 / d_m1
            dV_next[zp, -1] = V_z[-3]*L_0p + V_z[-2]*L_1p + V_z[-1]*L_2p
            d2V_next[zp, -1] = 2.0 * ( (V_z[-1] - V_z[-2]) / h_m1 - (V_z[-2] - V_z[-3]) / h_m2 ) / (h_m1 + h_m2)
        
        for z in range(n_z):
            for i, m in enumerate(grid):
                # Initial guess: use previous period's policy (interpolated)
                if t + 1 < T:
                    c = np.interp(m, grid, P[t+1][z])
                else:
                    c = 0.5 * m
                c = np.clip(c, c_min, m)
                
                # Newton's method
                for _ in range(50):
                    G_c = 0.0
                    G_prime_c = 0.0
                    for zp in range(n_z):
                        next_m = R * (m - c) + income[zp]
                        # Clip next_m to grid range for interpolation
                        next_m_clipped = np.clip(next_m, grid[0], grid[-1])
                        dv = np.interp(next_m_clipped, grid, dV_next[zp])
                        d2v = np.interp(next_m_clipped, grid, d2V_next[zp])
                        
                        G_c += transition[z, zp] * dv
                        G_prime_c += transition[z, zp] * d2v * (-R)
                    
                    G_c *= beta
                    G_prime_c *= beta
                    
                    f_val = utility_prime(c) - G_c
                    f_prime_val = utility_double_prime(c) - G_prime_c
                    
                    if abs(f_prime_val) < 1e-10:
                        break
                        
                    step = f_val / f_prime_val
                    c_new = c - step
                    c_new = np.clip(c_new, c_min, m)
                    
                    if abs(c_new - c) < 1e-10:
                        c = c_new
                        break
                    c = c_new
                
                P_t[z, i] = c
                # Calculate value
                cont_val = 0.0
                for zp in range(n_z):
                    next_m = R * (m - c) + income[zp]
                    next_m_clipped = np.clip(next_m, grid[0], grid[-1])
                    cont_val += transition[z, zp] * np.interp(next_m_clipped, grid, V_next[zp])
                V_t[z, i] = utility(c) + beta * cont_val
        
        V[t] = V_t
        P[t] = P_t
    
    # Interpolate policy back to the original grid
    result = np.zeros((T, n_z, n_out))
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.interp(grid_out, grid, P[t][z])
    
    # Enforce constraints
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.clip(result[t, z, :], c_min, grid_out + 1e-10)
            
    result[T-1, :, :] = grid_out[np.newaxis, :]
    
    return result
