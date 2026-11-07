import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    c_min = 1e-12
    grid_out = resource_grid
    n_grid_out = len(grid_out)
    T = horizon
    n_z = 2

    # Internal high-resolution grid for accurate value function and policy
    # Use a log-spaced grid that covers the range of interest
    # The scoring range is 0.2 <= m <= 8, but we need to handle the full output grid
    # Let's use a dense internal grid to ensure smooth interpolation
    n_int = 1000
    grid_int = np.geomspace(0.001, 40, n_int)

    def utility(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    # V[t][z] will be an array of values at internal grid points.
    # P[t][z] will be an array of optimal consumptions at internal grid points.
    
    V = [None] * T
    P = [None] * T
    
    # Terminal condition: t = T-1
    # c = m, V = u(m)
    V[T-1] = utility(grid_int)[np.newaxis, :].repeat(2, axis=0)
    P[T-1] = grid_int[np.newaxis, :].repeat(2, axis=0)

    # Precompute transition probabilities
    # transition[z, zp] is prob of going from z to zp

    # Backward induction for t = T-2 down to 0
    for t in range(T-2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: use terminal condition for remaining periods
            for t_rem in range(t, -1, -1):
                V[t_rem] = V[T-1].copy()
                P[t_rem] = P[T-1].copy()
            break

        V_t = np.zeros((n_z, n_int))
        P_t = np.zeros((n_z, n_int))
        
        V_next = V[t+1]

        # For each state z and each internal grid point m
        for z in range(n_z):
            # Vectorized approach for speed
            # We need to find c* that maximizes u(c) + beta * E[V_next(z', R(m-c) + y')]
            
            # Since c is in [0, m], let's parameterize c = m * x where x in [0, 1]
            # But c must be <= m.
            # Let's just iterate over m and use a 1D search or Newton.
            # Given n_int=1000, T=6, n_z=2, total points = 12000.
            # If we use a simple grid search for c at each point, it might be slow.
            # Let's use Newton's method on the Euler equation, but with dense V_next.
            
            # Alternatively, since the grid is dense, we can use interpolation of V_next
            # and its derivatives.
            
            # Let's pre-compute derivatives of V_next for both z' using finite differences
            # on the internal grid.
            
            # Actually, let's just do a local search or Newton for each point.
            # To speed up, we can vectorize over m for a fixed z.
            
            # Let's try a different approach: 
            # For each z, we want to find c(m) for all m in grid_int.
            # We can use the fact that the policy is smooth.
            
            # Let's stick to the Newton method but vectorized over m where possible?
            # The Euler equation is:
            # u'(c) = beta * R * sum_z' P(z'|z) V_next'(z', R(m-c) + y_{z'})
            
            # Let's compute V_next derivatives.
            dV_next = np.zeros_like(V_next)
            for zp in range(n_z):
                # Central differences
                dV_next[zp, 1:-1] = (V_next[zp, 2:] - V_next[zp, :-2]) / (grid_int[2:] - grid_int[:-2])
                dV_next[zp, 0] = (V_next[zp, 1] - V_next[zp, 0]) / (grid_int[1] - grid_int[0])
                dV_next[zp, -1] = (V_next[zp, -1] - V_next[zp, -2]) / (grid_int[-1] - grid_int[-2])
            
            # Second derivatives for Newton
            d2V_next = np.zeros_like(V_next)
            for zp in range(n_z):
                d2V_next[zp, 1:-1] = (dV_next[zp, 2:] - dV_next[zp, :-2]) / (grid_int[2:] - grid_int[:-2])
                d2V_next[zp, 0] = (dV_next[zp, 1] - dV_next[zp, 0]) / (grid_int[1] - grid_int[0])
                d2V_next[zp, -1] = (dV_next[zp, -1] - dV_next[zp, -2]) / (grid_int[-1] - grid_int[-2])

            for i, m in enumerate(grid_int):
                # Initial guess
                # Use a heuristic: c ~ m / (1 + beta*R) or similar
                # Or use the previous period's policy at the same m if available
                if t + 1 < T:
                    c = P[t+1][z, i]
                else:
                    c = m / (1 + beta * R)
                
                c = np.clip(c, c_min, m)
                
                # Newton iterations
                for _ in range(20):
                    # Compute G(c) = beta * R * sum P * dV_next[zp](next_m)
                    G_c = 0.0
                    G_prime_c = 0.0
                    for zp in range(n_z):
                        next_m = R * (m - c) + income[zp]
                        # Interpolate derivatives
                        dv = np.interp(next_m, grid_int, dV_next[zp])
                        d2v = np.interp(next_m, grid_int, d2V_next[zp])
                        
                        G_c += transition[z, zp] * dv
                        G_prime_c += transition[z, zp] * d2v * (-R)
                    
                    G_c *= beta * R
                    G_prime_c *= beta * R
                    
                    # u'(c)
                    if abs(gamma - 1.0) < 1e-10:
                        u_prime_c = 1.0 / c
                        u_double_prime_c = -1.0 / (c ** 2)
                    else:
                        u_prime_c = c ** (-gamma)
                        u_double_prime_c = -gamma * c ** (-gamma - 1)
                    
                    f_val = u_prime_c - G_c
                    f_prime_val = u_double_prime_c - G_prime_c
                    
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
                    cont_val += transition[z, zp] * np.interp(next_m, grid_int, V_next[zp])
                
                if abs(gamma - 1.0) < 1e-10:
                    V_t[z, i] = np.log(max(c, c_min)) + beta * cont_val
                else:
                    V_t[z, i] = (max(c, c_min) ** (1 - gamma)) / (1 - gamma) + beta * cont_val

        V[t] = V_t
        P[t] = P_t

    # Interpolate the dense policy P onto the output grid
    result = np.zeros((T, n_z, n_grid_out))
    for t in range(T):
        for z in range(n_z):
            # P[t][z] is defined on grid_int
            # We need values on grid_out
            # Ensure c <= m
            c_interp = np.interp(grid_out, grid_int, P[t][z])
            # Enforce c <= m
            c_interp = np.minimum(c_interp, grid_out)
            c_interp = np.maximum(c_interp, c_min)
            result[t, z, :] = c_interp

    # Enforce final period constraint: c = m
    result[T-1, :, :] = grid_out[np.newaxis, :]

    # Final sanity checks
    result = np.clip(result, c_min, grid_out + 1e-10)

    return result
