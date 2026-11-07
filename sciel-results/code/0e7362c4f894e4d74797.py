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

    # We will store the policy function as a function of m for each z and t.
    # Since the problem is small (101 grid points, 2 states, 6 periods), 
    # we can solve for the optimal c at each grid point using Newton's method.
    
    # V[t][z] will be an array of values at grid points.
    # P[t][z] will be an array of optimal consumptions at grid points.
    
    V = [None] * T
    P = [None] * T
    
    # Terminal condition: t = T-1
    # c = m, V = u(m)
    V[T-1] = utility(grid)[np.newaxis, :].repeat(2, axis=0)
    P[T-1] = grid[np.newaxis, :].repeat(2, axis=0)
    
    # Backward induction for t = T-2 down to 0
    for t in range(T-2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: use a simple rule if we run out of time
            # This shouldn't happen with Newton's method
            pass
            
        V_t = np.zeros((n_z, n_grid))
        P_t = np.zeros((n_z, n_grid))
        
        # V_next is V[t+1]
        V_next = V[t+1]
        
        # For each z and m in grid:
        # Maximize u(c) + beta * sum_{z'} P(z'|z) V_next[z'](R(m-c) + y_{z'})
        # FOC: u'(c) = beta * sum_{z'} P(z'|z) V_next'[z'](R(m-c) + y_{z'}) * R
        
        for z in range(n_z):
            for i, m in enumerate(grid):
                # We need to solve for c in (0, m]
                # Define F(c) = u'(c) - beta * R * sum_{z'} P(z'|z) V_next'[z'](R(m-c) + y_{z'})
                # We need F(c) = 0
                
                # To compute V_next'[z'](x), we need to interpolate V_next[z'] and differentiate.
                # Since V_next is convex (usually), we can use linear interpolation of the derivative
                # or just evaluate the derivative of the interpolated function.
                # Actually, for CRRA, the value function is often close to a power function or log.
                # But let's stick to the grid.
                
                # We can approximate V_next'[z'](x) by the slope of the secant line between grid points,
                # or better, since we have the grid, we can use np.gradient or just finite differences.
                # However, a more robust way for this specific task is to note that the value function
                # is smooth. We can pre-compute the derivative of V_next on the grid.
                
                # Let's pre-compute derivatives of V_next for both z'
                # Using central differences where possible, one-sided at edges.
                dV_next = np.zeros_like(V_next)
                for zp in range(n_z):
                    dV_next[zp, 1:-1] = (V_next[zp, 2:] - V_next[zp, :-2]) / (grid[2:] - grid[:-2])
                    dV_next[zp, 0] = (V_next[zp, 1] - V_next[zp, 0]) / (grid[1] - grid[0])
                    dV_next[zp, -1] = (V_next[zp, -1] - V_next[zp, -2]) / (grid[-1] - grid[-2])
                
                # Now, for a given c, the continuation derivative is:
                # G(c) = beta * R * sum_{z'} P(z'|z) * interp(dV_next[z'], R(m-c) + y_{z'})
                
                # We want to find c such that u'(c) = G(c).
                
                # Newton's method:
                # c_{k+1} = c_k - (u'(c_k) - G(c_k)) / (u''(c_k) - G'(c_k))
                
                # G'(c) = beta * R * sum_{z'} P(z'|z) * dV_next'[z'](R(m-c) + y_{z'}) * (-R)
                # dV_next'[z'](x) is the second derivative of V_next.
                # We can approximate dV_next' similarly with finite differences of dV_next.
                
                d2V_next = np.zeros_like(V_next)
                for zp in range(n_z):
                    d2V_next[zp, 1:-1] = (dV_next[zp, 2:] - dV_next[zp, :-2]) / (grid[2:] - grid[:-2])
                    d2V_next[zp, 0] = (dV_next[zp, 1] - dV_next[zp, 0]) / (grid[1] - grid[0])
                    d2V_next[zp, -1] = (dV_next[zp, -1] - dV_next[zp, -2]) / (grid[-1] - grid[-2])
                
                # Initial guess for c:
                # In steady state, c is roughly proportional to m.
                # Let's try c = 0.5 * m as a start, or based on beta.
                # A better heuristic: c* = (beta * R)^(-1/gamma) * (m + y_bar / (R-1)) ... complicated.
                # Let's just use c = 0.5 * m, clipped to [c_min, m].
                
                c = 0.5 * m
                c = np.clip(c, c_min, m)
                
                for _ in range(20):
                    # Compute G(c)
                    G_c = 0.0
                    G_prime_c = 0.0
                    for zp in range(n_z):
                        next_m = R * (m - c) + income[zp]
                        # Interpolate dV_next and d2V_next
                        dv = np.interp(next_m, grid, dV_next[zp])
                        d2v = np.interp(next_m, grid, d2V_next[zp])
                        
                        G_c += transition[z, zp] * dv
                        G_prime_c += transition[z, zp] * d2v * (-R)
                    
                    G_c *= beta * R
                    G_prime_c *= beta * R
                    # Note: G(c) = beta * R * sum P * dV. 
                    # The term in FOC is beta * R * sum P * dV.
                    # Wait, the FOC is u'(c) = beta * sum P * V'(next_m) * R.
                    # So G(c) = beta * R * sum P * dV_next[zp](next_m).
                    # And G'(c) = beta * R * sum P * d2V_next[zp](next_m) * (-R) = -beta * R^2 * sum P * d2V.
                    
                    # Let's re-verify signs.
                    # V'(next_m) where next_m = R(m-c) + y.
                    # d/dc V'(next_m) = V''(next_m) * (-R).
                    # So G'(c) = beta * R * sum P * V''(next_m) * (-R) = -beta * R^2 * sum P * V''(next_m).
                    
                    # My G_prime_c calculation above:
                    # G_prime_c = beta * R * sum P * d2v * (-R) = -beta * R^2 * sum P * d2v. Correct.
                    
                    f_val = utility_prime(c) - G_c
                    f_prime_val = utility_double_prime(c) - G_prime_c
                    
                    if abs(f_prime_val) < 1e-12:
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
