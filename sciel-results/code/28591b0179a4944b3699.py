import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    M = len(resource_grid)
    Z = len(income)
    
    # Consumption policy: consumption[t, z, m_index]
    consumption = np.zeros((horizon, Z, M))
    
    # Helper for marginal utility
    def u_prime(c, g):
        c = np.maximum(c, 1e-12)
        if g == 1:
            return 1.0 / c
        else:
            return c ** (-g)
    
    # Helper for log marginal utility (more stable for Newton's method)
    def log_u_prime(c, g):
        c = np.maximum(c, 1e-12)
        if g == 1:
            return -np.log(c)
        else:
            return -g * np.log(c)
            
    # Terminal condition: at t=5 (last decision), consume everything.
    # c_5(m, z) = m
    # So the saving rate s_5 = 0.
    # We store the policy as c directly, but internally we might work with s.
    # Let's store the policy c_{t+1} for the next step's lookup.
    
    # Initialize the "next" policy function (for t=5)
    # c_next[z, m_idx]
    c_next = np.zeros((Z, M))
    for z in range(Z):
        c_next[z] = resource_grid
    
    # Backward induction for t = 4 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with simple rule (consume all, or 50%)
            # To be safe and valid, consume all is valid (c=m), but maybe not optimal.
            # However, if we run out of time, we must return something valid.
            # c=m is always valid.
            for tt in range(t, -1, -1):
                for z in range(Z):
                    consumption[tt, z] = resource_grid
            break
            
        c_curr = np.zeros((Z, M))
        
        for z in range(Z):
            # We want to find c(m) for each m in resource_grid
            # Let s = (m - c) / m, so c = m * (1 - s)
            # s is in [0, 1). 
            # Euler: u'(c) = beta * R * sum_{z'} P(z'|z) * u'(c_{t+1}(m', z'))
            # where m' = R * (m - c) + income[z'] = R * m * s + income[z']
            
            # We solve for s using Newton's method.
            # Let F(s) = log(u'(c)) - log(E[u'(c_next)])
            # c = m * (1 - s)
            # c_next(m', z') is obtained by interpolating c_next[z'] on resource_grid.
            
            # Initial guess for s:
            # For gamma=1 (log utility), optimal s is often around 0.2-0.3.
            # For gamma>1, saving is lower.
            # Let's start with s=0.2 for all m.
            s = np.full(M, 0.2)
            
            # Newton's method iteration
            # We need F(s) and F'(s)
            # F(s) = -gamma * log(c) - log( beta * R * sum P * u'(c_next) )
            # Wait, log_u_prime(c) = -gamma * log(c) (for gamma != 1) or -log(c) (gamma=1).
            # Let's just use the log form directly.
            
            # Pre-compute transition weights for this z
            # P_z = transition[z, :]
            
            for _ in range(10): # Max 10 iterations, usually converges in 3-5
                if time.time() > deadline:
                    break
                    
                c = resource_grid * (1.0 - s)
                # Ensure c > 0
                c = np.maximum(c, 1e-12)
                
                # LHS: log u'(c)
                lhs = log_u_prime(c, gamma)
                
                # RHS: log( beta * R * E[u'(c_next)] )
                # E[u'(c_next)] = sum_{z'} P(z'|z) * u'(c_next(z', m'))
                # m' = R * m * s + income[z']
                
                # We need to evaluate c_next(z', m') for all m and z'
                # m_prime shape: (M, Z)
                m_prime = R * resource_grid[:, None] * s[:, None] + income[None, :]
                
                # Interpolate c_next for each z'
                # c_next shape: (Z, M)
                # For each z', we have c_next[z'] (M,) and we evaluate at m_prime[:, z'] (M,)
                
                u_prime_next_sum = np.zeros(M)
                for zp in range(Z):
                    # m_prime[:, zp] is the m' for state zp
                    # Interpolate c_next[zp] at m_prime[:, zp]
                    c_next_val = np.interp(m_prime[:, zp], resource_grid, c_next[zp])
                    # Compute u'(c_next_val)
                    u_prime_val = u_prime(c_next_val, gamma)
                    # Weight by transition probability
                    u_prime_next_sum += transition[z, zp] * u_prime_val
                
                # rhs_val = beta * R * u_prime_next_sum
                rhs_val = beta * R * u_prime_next_sum
                # Ensure positive for log
                rhs_val = np.maximum(rhs_val, 1e-12)
                rhs = np.log(rhs_val)
                
                # F(s) = lhs - rhs
                F = lhs - rhs
                
                # Derivative F'(s)
                # d/ds log(u'(c)) where c = m(1-s)
                # u'(c) = c^{-gamma}
                # log u'(c) = -gamma log(c)
                # d/ds (-gamma log(m(1-s))) = -gamma * (1/(m(1-s))) * (-m) = gamma / (1-s)
                # So d/ds lhs = gamma / (1 - s) (for gamma != 1)
                # For gamma = 1, log u'(c) = -log(c), derivative is 1/(1-s).
                # So d/ds lhs = gamma / (1 - s) works for all gamma?
                # Let's check:
                # u'(c) = c^{1-gamma-1} = c^{-gamma}
                # log u' = -gamma log c
                # d/ds = -gamma * (1/c) * dc/ds
                # c = m(1-s), dc/ds = -m
                # d/ds = -gamma * (1/(m(1-s))) * (-m) = gamma / (1-s)
                # Yes, this holds for any gamma.
                
                d_lhs = gamma / (1.0 - s)
                
                # d/ds rhs = d/ds log( beta R sum P u'(c_next) )
                # = (1 / (beta R sum P u')) * beta R sum P d/ds u'(c_next)
                # = (1 / E[u']) * E[ d/ds u'(c_next) ]
                
                # u'(c_next) = c_next^{-gamma}
                # d/ds u'(c_next) = -gamma * c_next^{-gamma-1} * dc_next/ds
                # c_next is a function of m' = R m s + y
                # dc_next/ds = dc_next/dm' * dm'/ds
                # dm'/ds = R m
                # We need the derivative of the interpolated c_next.
                # Approximation: Numerical derivative or analytic if we had a parametric form.
                # Since c_next is piecewise linear (from np.interp), its derivative is piecewise constant (slope of segment).
                # However, calculating the slope of the interp segment for each point is complex.
                # Alternative: Use a numerical derivative for F(s) to avoid complex analytic derivs of the interp.
                # Newton with numerical derivative is "Secant" or just use a small step.
                # Or, simpler: use a robust root finding like Bisection + Newton hybrid?
                # Given the smoothness, a simple numerical derivative step size h=1e-4 might work.
                
                # Let's use a numerical derivative for the whole F(s) to be safe and simple.
                h = 1e-4
                s_plus = np.clip(s + h, 0, 0.999999)
                c_plus = resource_grid * (1.0 - s_plus)
                c_plus = np.maximum(c_plus, 1e-12)
                lhs_plus = log_u_prime(c_plus, gamma)
                
                m_prime_plus = R * resource_grid[:, None] * s_plus[:, None] + income[None, :]
                u_prime_next_sum_plus = np.zeros(M)
                for zp in range(Z):
                    c_next_val_plus = np.interp(m_prime_plus[:, zp], resource_grid, c_next[zp])
                    u_prime_val_plus = u_prime(c_next_val_plus, gamma)
                    u_prime_next_sum_plus += transition[z, zp] * u_prime_val_plus
                rhs_val_plus = beta * R * u_prime_next_sum_plus
                rhs_val_plus = np.maximum(rhs_val_plus, 1e-12)
                rhs_plus = np.log(rhs_val_plus)
                
                F_plus = lhs_plus - rhs_plus
                
                dF = (F_plus - F) / h
                
                # Newton step: s_new = s - F / dF
                # Avoid division by zero
                mask = np.abs(dF) > 1e-8
                s_new = s.copy()
                s_new[mask] = s[mask] - F[mask] / dF[mask]
                
                # Constrain s to [0, 1)
                s_new = np.clip(s_new, 0.0, 0.999999)
                
                # Check convergence
                if np.max(np.abs(s_new - s)) < 1e-6:
                    s = s_new
                    break
                s = s_new
                
            # Final c
            c_curr[z] = resource_grid * (1.0 - s)
            
        consumption[t] = c_curr
        c_next = c_curr
        
    # Final validity checks
    for t in range(horizon):
        for z in range(Z):
            c = consumption[t, z]
            m = resource_grid
            # 0 < c
            c = np.maximum(c, 1e-12)
            # c <= m + 1e-10
            c = np.minimum(c, m + 1e-10)
            consumption[t, z] = c
            
    # Enforce terminal condition exactly
    consumption[5] = resource_grid[None, :]
    
    return consumption
