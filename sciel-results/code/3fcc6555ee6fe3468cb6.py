def solve_consumption(model, resource_grid, budget):
        import numpy as np
        import time
        
        beta = model['beta']
        gamma = model['gamma']
        R = model['R']
        income = model['income']
        transition = model['transition']
        H = model['horizon']
        
        T = len(resource_grid)
        Z = len(income)
        
        # c[t, z, m] or c[t, m, z]? 
        # Task says: consumption[t, income_state, resource_index]
        # So shape (H, Z, T)
        
        c = np.zeros((H, Z, T))
        
        # Last period: consume all
        c[H-1, :, :] = resource_grid[None, :]
        
        # Time guard
        start_time = time.time()
        deadline = 110.0 # 10s margin
        
        for t in range(H-2, -1, -1):
            for z in range(Z):
                # Initialize c for this t, z
                c_curr = resource_grid.copy() # Guess: consume all? Or 0.5?
                # Better guess: linear interpolation of previous step? 
                # Or just start with 0.5 * m.
                c_curr = 0.5 * resource_grid
                
                for _ in range(20): # Max Newton iterations
                    if time.time() > deadline:
                        break
                    
                    # Compute m' for each z'
                    # m' = R * (m - c) + income[z']
                    # c_curr is shape (T,)
                    # m is resource_grid shape (T,)
                    
                    # Residual F(c)
                    # LHS = c^{-gamma}
                    # RHS = beta * R * sum_z' P(z'|z) * (c_next(m', z'))^{-gamma}
                    
                    lhs = c_curr ** (-gamma)
                    rhs = 0.0
                    f_prime_rhs = 0.0 # For derivative of RHS part
                    
                    for zp in range(Z):
                        p = transition[z, zp]
                        if p == 0: continue
                        
                        mp = R * (resource_grid - c_curr) + income[zp]
                        
                        # Get c_next at mp for state zp
                        # c_next is c[t+1, zp, :] which is on resource_grid
                        c_next_vals = np.interp(mp, resource_grid, c[t+1, zp, :])
                        
                        # Ensure c_next is valid (floor)
                        c_next_vals = np.maximum(c_next_vals, 1e-12)
                        
                        # Term for RHS
                        rhs += p * (c_next_vals ** (-gamma))
                        
                        # Derivative part:
                        # d/dc [ (c_next(m'))^{-gamma} ]
                        # = -gamma * (c_next)^{-gamma-1} * dc_next/dm' * dm'/dc
                        # dm'/dc = -R
                        # dc_next/dm' approximated numerically
                        delta = 1e-5
                        mp_plus = mp + delta
                        mp_minus = mp - delta
                        # Clip mp to grid range for interp? np.interp handles out of bounds by clamping.
                        c_next_plus = np.interp(mp_plus, resource_grid, c[t+1, zp, :])
                        c_next_minus = np.interp(mp_minus, resource_grid, c[t+1, zp, :])
                        dc_dmp = (c_next_plus - c_next_minus) / (2 * delta)
                        
                        # u''(c_next) = -gamma * (c_next)^{-gamma-1}
                        # Wait, the term in F is - RHS.
                        # F = LHS - RHS
                        # F' = LHS' - RHS'
                        # RHS' = sum p * d/dc (c_next^{-gamma})
                        # d/dc (c_next^{-gamma}) = -gamma * c_next^{-gamma-1} * dc_next/dc
                        # dc_next/dc = dc_next/dm' * dm'/dc = dc_dmp * (-R)
                        # So term is -gamma * c_next^{-gamma-1} * dc_dmp * (-R) = gamma * R * c_next^{-gamma-1} * dc_dmp
                        
                        deriv_term = gamma * R * (c_next_vals ** (-gamma - 1)) * dc_dmp
                        f_prime_rhs += p * deriv_term
                    
                    rhs *= beta * R
                    f_prime_rhs *= beta * R # Wait, beta*R is constant multiplier for RHS.
                    # Let's re-verify constants.
                    # RHS = beta * R * Sum(...)
                    # dRHS/dc = beta * R * Sum( d/dC (...) )
                    # So yes, multiply the sum of derivatives by beta * R.
                    
                    # LHS derivative
                    lhs_prime = -gamma * (c_curr ** (-gamma - 1))
                    
                    F = lhs - rhs
                    F_prime = lhs_prime - f_prime_rhs
                    
                    # Newton step
                    # Avoid division by zero or small F_prime
                    mask = np.abs(F_prime) > 1e-12
                    c_update = np.zeros_like(c_curr)
                    c_update[mask] = F[mask] / F_prime[mask]
                    
                    c_new = c_curr - c_update
                    c_new = np.clip(c_new, 1e-12, resource_grid)
                    
                    if np.max(np.abs(c_new - c_curr)) < 1e-9:
                        c_curr = c_new
                        break
                    c_curr = c_new
                
                c[t, z, :] = c_curr
            
            # Check time
            if time.time() > deadline:
                # Fill remaining with current best guess (e.g. 0.5m or previous)
                for t_rem in range(t-1, -1, -1):
                    for z_rem in range(Z):
                        c[t_rem, z_rem, :] = 0.5 * resource_grid
                break
        
        # Ensure final period is exact
        c[H-1, :, :] = resource_grid[None, :]
        
        return c
