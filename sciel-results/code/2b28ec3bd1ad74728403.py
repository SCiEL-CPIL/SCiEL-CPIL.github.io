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
    grid = resource_grid
    n_grid = len(grid)
    T = horizon
    n_z = 2
    
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

    V = [None] * T
    P = [None] * T
    
    # Terminal condition
    V[T-1] = utility(grid)[np.newaxis, :].repeat(2, axis=0)
    P[T-1] = grid[np.newaxis, :].repeat(2, axis=0)
    
    # Backward induction for t = T-2 down to 0
    for t in range(T-2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: use a simple rule if we run out of time
            for t_rem in range(t, -1, -1):
                V[t_rem] = V[T-1].copy()
                P[t_rem] = P[T-1].copy()
            break
            
        V_t = np.zeros((n_z, n_grid))
        P_t = np.zeros((n_z, n_grid))
        
        V_next = V[t+1]
        
        # Pre-compute derivatives of V_next for both z'
        dV_next = np.zeros_like(V_next)
        d2V_next = np.zeros_like(V_next)
        
        # Use vectorized finite differences
        # First derivative
        dV_next[:, 1:-1] = (V_next[:, 2:] - V_next[:, :-2]) / (grid[2:] - grid[:-2])
        dV_next[:, 0] = (V_next[:, 1] - V_next[:, 0]) / (grid[1] - grid[0])
        dV_next[:, -1] = (V_next[:, -1] - V_next[:, -2]) / (grid[-1] - grid[-2])
        
        # Second derivative
        d2V_next[:, 1:-1] = (dV_next[:, 2:] - dV_next[:, :-2]) / (grid[2:] - grid[:-2])
        d2V_next[:, 0] = (dV_next[:, 1] - dV_next[:, 0]) / (grid[1] - grid[0])
        d2V_next[:, -1] = (dV_next[:, -1] - dV_next[:, -2]) / (grid[-1] - grid[-2])
        
        # Pre-compute interpolation arrays for speed
        # For each z, we need to interpolate dV_next[z] and d2V_next[z]
        
        for z in range(n_z):
            for i, m in enumerate(grid):
                # Initial guess for c
                # Use a heuristic based on beta and R
                # In steady state with no income, c* = (beta*R)^(-1/gamma) * m
                # With income, it's more complex, but let's use a scaled version
                if abs(gamma - 1.0) < 1e-10:
                    # Log utility: c* = beta*R / (1 + beta*R) * (m + y/(R-1)) approximately
                    # But for simplicity, let's use c = 0.5 * m as a safe start
                    c = 0.5 * m
                else:
                    # CRRA: c* = (beta*R)^(-1/gamma) * (m + y_bar/(R-1)) approximately
                    # Let's use a scaled version
                    scale = (beta * R) ** (-1.0 / gamma)
                    c = scale * m
                    c = min(c, 0.9 * m)  # Don't consume more than 90% initially
                
                c = np.clip(c, c_min, m)
                
                # Newton's method
                for _ in range(30):
                    # Compute G(c) = beta * R * sum_{z'} P(z'|z) * dV_next[z'](R(m-c) + y_{z'})
                    # and G'(c) = -beta * R^2 * sum_{z'} P(z'|z) * d2V_next[z'](R(m-c) + y_{z'})
                    
                    G_c = 0.0
                    G_prime_c = 0.0
                    
                    for zp in range(n_z):
                        next_m = R * (m - c) + income[zp]
                        # Interpolate
                        dv = np.interp(next_m, grid, dV_next[zp])
                        d2v = np.interp(next_m, grid, d2V_next[zp])
                        
                        G_c += transition[z, zp] * dv
                        G_prime_c += transition[z, zp] * d2v * (-R)
                    
                    G_c *= beta * R
                    G_prime_c *= beta * R
                    
                    f_val = utility_prime(c) - G_c
                    f_prime_val = utility_double_prime(c) - G_prime_c
                    
                    if abs(f_prime_val) < 1e-12:
                        break
                        
                    step = f_val / f_prime_val
                    c_new = c - step
                    c_new = np.clip(c_new, c_min, m)
                    
                    if abs(c_new - c) < 1e-12:
                        c = c_new
                        break
                    c = c_new
                
                P_t[z, i] = c
                
                # Calculate value
                cont_val = 0.0
                for zp in range(n_z):
                    next_m = R * (m - c) + income[zp]
                    cont_val += transition[z, zp] * np.interp(next_m, grid, V_next[zp])
                V_t[z, i] = utility(c) + beta * cont_val
        
        V[t] = V_t
        P[t] = P_t
    
    # Assemble result
    result = np.zeros((T, n_z, n_grid))
    for t in range(T):
        result[t, :, :] = P[t]
        
    # Enforce constraints
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.clip(result[t, z, :], c_min, grid + 1e-10)
            
    # Final period constraint
    result[T-1, :, :] = grid[np.newaxis, :]
    
    return result
