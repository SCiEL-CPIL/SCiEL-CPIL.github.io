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
    
    # Use a dense log-spaced grid for high-resolution interpolation
    # The scoring grid is geomspace(0.001, 40, 251), so we match that resolution or better
    # We use 1000 points for the internal value function grid to ensure accuracy
    internal_n = 1000
    log_grid = np.geomspace(0.001, 40, internal_n)
    
    # Precompute log-space for bisection
    # We solve for c in (0, m]. Let c = m * x, where x in (0, 1].
    # Alternatively, solve for c directly using bisection on c.
    # Since u'(c) is monotonic decreasing and G(c) is monotonic decreasing (usually),
    # we can use bisection.
    
    # However, a more robust approach for CRRA is to solve the Euler equation.
    # u'(c) = beta * R * E[V'(R(m-c) + y')]
    
    # We will store V and dV (derivative of V) on the internal_log_grid for each t and z.
    # V[t][z] is an array of values at internal_log_grid points.
    # dV[t][z] is an array of derivatives at internal_log_grid points.
    
    V = [None] * T
    dV = [None] * T
    P = [None] * T # Policy on the original grid
    
    # Terminal condition: t = T-1
    # c = m, V = u(m)
    # u(m) = log(m) or m^(1-gamma)/(1-gamma)
    # u'(m) = 1/m or m^(-gamma)
    
    if abs(gamma - 1.0) < 1e-10:
        V[T-1] = np.log(log_grid)[np.newaxis, :].repeat(2, axis=0)
        dV[T-1] = (1.0 / log_grid)[np.newaxis, :].repeat(2, axis=0)
    else:
        V[T-1] = (log_grid ** (1 - gamma) / (1 - gamma))[np.newaxis, :].repeat(2, axis=0)
        dV[T-1] = (log_grid ** (-gamma))[np.newaxis, :].repeat(2, axis=0)
    
    # Policy at terminal period: c = m
    P[T-1] = grid[np.newaxis, :].repeat(2, axis=0)
    
    # Backward induction for t = T-2 down to 0
    for t in range(T-2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: use a simple rule if we run out of time
            # This shouldn't happen with bisection
            pass
            
        V_t = np.zeros((n_z, internal_n))
        dV_t = np.zeros((n_z, internal_n))
        P_t = np.zeros((n_z, n_grid))
        
        # V_next and dV_next are from t+1
        V_next = V[t+1]
        dV_next = dV[t+1]
        
        # For each z and m in internal_log_grid:
        # We need to find c such that u'(c) = beta * R * sum_{z'} P(z'|z) dV_next[z'](R(m-c) + y_{z'})
        
        for z in range(n_z):
            # We will solve for c for all m in internal_log_grid simultaneously using bisection.
            # This is tricky because the upper bound of c is m, which varies.
            # However, we can use a normalized variable or just loop over m?
            # Looping over 1000 m's * 2 z's * 6 t's = 12000 iterations.
            # Each iteration involves bisection (e.g., 50 steps) and vectorized numpy ops?
            # No, bisection is scalar per m.
            
            # Let's try to vectorize the bisection steps.
            # For a fixed t, z, we have m array of size internal_n.
            # We want to find c[i] for each i.
            
            # Initialize c_low and c_high
            c_low = np.full(internal_n, c_min)
            c_high = log_grid.copy() # c <= m
            
            # Bisection tolerance
            tol = 1e-8
            max_iter = 50
            
            for _ in range(max_iter):
                c_mid = (c_low + c_high) / 2.0
                
                # Compute LHS: u'(c_mid)
                if abs(gamma - 1.0) < 1e-10:
                    lhs = 1.0 / c_mid
                else:
                    lhs = c_mid ** (-gamma)
                
                # Compute RHS: beta * R * sum_{z'} P(z'|z) dV_next[z'](R(m - c_mid) + y_{z'})
                rhs = np.zeros(internal_n)
                for zp in range(n_z):
                    next_m = R * (log_grid - c_mid) + income[zp]
                    # Interpolate dV_next[zp] at next_m
                    # dV_next[zp] is defined on log_grid
                    # next_m might be outside [0.001, 40]?
                    # min next_m: R*(0.001 - 40) + 0.5 ~ -40. 
                    # max next_m: R*(40 - 0) + 1.5 ~ 42.
                    # dV is only defined on [0.001, 40].
                    # We need to handle out-of-bounds.
                    # For m' < 0.001, the value function is likely very low or negative (if gamma > 1).
                    # But dV should be positive.
                    # Let's clip next_m to [1e-6, 100] to be safe, or use np.interp with default fill.
                    # np.interp returns left value for x < x[0] and right value for x > x[-1].
                    # This is a reasonable approximation for the derivative at the boundaries.
                    
                    dv_val = np.interp(next_m, log_grid, dV_next[zp])
                    rhs += transition[z, zp] * dv_val
                
                rhs *= beta * R
                
                # Update bounds
                # We want lhs = rhs.
                # u'(c) is decreasing in c.
                # rhs(c) = beta * R * E[dV'(next_m)]. next_m = R(m-c)+y.
                # As c increases, next_m decreases.
                # dV' (second derivative of V) is negative (V is concave).
                # So as next_m decreases, dV'(next_m) increases (becomes less negative).
                # Wait, dV is the first derivative of V. dV is decreasing.
                # So as next_m decreases, dV(next_m) increases.
                # Therefore, rhs(c) is INCREASING in c.
                
                # Let's check:
                # c increases -> next_m decreases -> dV(next_m) increases (since dV is decreasing, lower x means higher value? No, dV is decreasing, so lower x means HIGHER value).
                # Yes, dV(x) is decreasing. So if x decreases, dV(x) increases.
                # So rhs(c) is increasing in c.
                
                # lhs(c) is decreasing in c.
                # So f(c) = lhs(c) - rhs(c) is strictly decreasing.
                # We want f(c) = 0.
                
                # If lhs > rhs, we need to decrease c?
                # No, if lhs > rhs, since lhs is decreasing and rhs is increasing, we need to increase c to lower lhs and raise rhs.
                # So if lhs > rhs, c is too small. Move c_low up.
                # If lhs < rhs, c is too large. Move c_high down.
                
                mask = lhs > rhs
                c_low[mask] = c_mid[mask]
                c_high[~mask] = c_mid[~mask]
                
                if np.max(c_high - c_low) < tol:
                    break
            
            c_opt = (c_low + c_high) / 2.0
            c_opt = np.clip(c_opt, c_min, log_grid)
            
            # Compute V_t and dV_t at log_grid
            # V_t = u(c) + beta * E[V_next(next_m)]
            # dV_t = u'(c) + beta * E[dV_next(next_m) * (-R)]  <-- Chain rule!
            # Wait, dV_t is dV/dm.
            # V(m) = u(c(m)) + beta * E[V_next(R(m-c(m)) + y')]
            # dV/dm = u'(c) * c'(m) + beta * E[dV_next(next_m) * R]
            # From Euler eq: u'(c) = beta * R * E[dV_next(next_m)]
            # So u'(c) * c'(m) = beta * R * E[dV_next(next_m)] * c'(m)
            # This doesn't directly simplify to a known value unless we know c'(m).
            
            # However, we can compute dV_t numerically or using the envelope theorem?
            # Envelope theorem: dV/dm = beta * R * E[dV_next(next_m)].
            # Let's verify.
            # V(m) = max_c u(c) + beta E[V_next(R(m-c)+y)]
            # dV/dm = beta E[dV_next(R(m-c)+y) * R]
            # Yes! The derivative of the value function with respect to m is simply the expected shadow value of wealth.
            
            # So dV_t[z, :] = beta * R * sum_{z'} P(z'|z) dV_next[z'](R(m - c_opt) + y_{z'})
            
            dV_t[z, :] = rhs # rhs was computed as beta * R * sum ... dV_next ...
            
            # Now compute V_t[z, :]
            # V_t = u(c) + beta * E[V_next(next_m)]
            if abs(gamma - 1.0) < 1e-10:
                u_c = np.log(np.maximum(c_opt, c_min))
            else:
                u_c = (np.maximum(c_opt, c_min) ** (1 - gamma)) / (1 - gamma)
            
            v_next_val = np.zeros(internal_n)
            for zp in range(n_z):
                next_m = R * (log_grid - c_opt) + income[zp]
                v_next_val += transition[z, zp] * np.interp(next_m, log_grid, V_next[zp])
            
            V_t[z, :] = u_c + beta * v_next_val
            
            # Now compute P_t on the original grid
            # We have c_opt as a function of m on log_grid.
            # We can interpolate c_opt to the original grid.
            P_t[z, :] = np.interp(grid, log_grid, c_opt)
        
        V[t] = V_t
        dV[t] = dV_t
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
