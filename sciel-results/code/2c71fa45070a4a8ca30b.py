import numpy as np

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    N = len(resource_grid)
    T = horizon
    Z = len(income)
    
    # Policy array: c[t, z, i]
    c_policy = np.zeros((T, Z, N))
    
    # Last period: consume everything
    c_policy[T-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    def util_prime(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return 1.0 / c
        else:
            return c ** (-gamma)
            
    # Backward induction
    for t in range(T-2, -1, -1):
        # c_next is c_policy[t+1]
        c_next = c_policy[t+1]
        
        for z in range(Z):
            for i in range(N):
                m = resource_grid[i]
                
                # Objective derivative function for bisection
                # f(c) = U'(c) - beta * E[U'(c') * R]
                # c' = c_policy[t+1][z', index_of(m')]
                # m' = R * (m - c) + income[z']
                
                def f(c_val):
                    # c_val is scalar
                    u_prime_c = util_prime(c_val)
                    
                    exp_term = 0.0
                    for zp in range(Z):
                        prob = transition[z, zp]
                        if prob < 1e-15:
                            continue
                        m_prime = R * (m - c_val) + income[zp]
                        # Interpolate c_next[zp, :] at m_prime
                        # c_next[zp] is the policy for state zp at t+1
                        # resource_grid is the x-axis
                        c_prime = np.interp(m_prime, resource_grid, c_next[zp])
                        u_prime_cp = util_prime(c_prime)
                        exp_term += prob * R * u_prime_cp
                    
                    return u_prime_c - beta * exp_term
                
                # Bisection for f(c) = 0
                # f is decreasing
                # f(1e-12) > 0 (U'(c) huge, exp term finite)
                # f(m) < 0 (U'(m) small, exp term usually larger? 
                # Wait, if c=m, m'=income. 
                # If c=0, m'=R*m+income. 
                # Generally, optimal c is in between.
                # We need to ensure f(lo) > 0 and f(hi) < 0.
                
                lo = 1e-12
                hi = m
                
                # Check bounds
                f_lo = f(lo)
                f_hi = f(hi)
                
                # Due to numerical issues or specific parameters, signs might be unexpected
                # But theoretically f(lo) > 0 and f(hi) < 0 for concave utility.
                # If f(hi) > 0, it means we should consume more than m? Impossible.
                # If f(lo) < 0, it means we should consume less than lo? 
                
                # Let's just clamp if necessary, but standard bisection assumes sign change.
                if f_lo * f_hi > 0:
                    # Fallback to grid search if bisection fails
                    # This is rare
                    c_candidates = np.linspace(1e-12, m, 100)
                    vals = np.array([f(c) for c in c_candidates])
                    # We want f(c)=0. Since f is decreasing, we want the sign change.
                    # If all positive, pick max c. If all negative, pick min c.
                    if np.all(vals > 0):
                        c_opt = m
                    elif np.all(vals < 0):
                        c_opt = 1e-12
                    else:
                        idx = np.where(np.diff(np.sign(vals)) != 0)[0]
                        if len(idx) > 0:
                            i_idx = idx[0]
                            c1 = c_candidates[i_idx]
                            c2 = c_candidates[i_idx+1]
                            # Refine with bisection on this small interval
                            lo_b, hi_b = c1, c2
                            for _ in range(20):
                                mid = (lo_b + hi_b) / 2
                                if f(mid) > 0:
                                    lo_b = mid
                                else:
                                    hi_b = mid
                            c_opt = (lo_b + hi_b) / 2
                        else:
                            c_opt = m / 2
                else:
                    for _ in range(50):
                        mid = (lo + hi) / 2
                        f_mid = f(mid)
                        if f_mid > 0:
                            lo = mid
                        else:
                            hi = mid
                    c_opt = (lo + hi) / 2
                
                c_policy[t, z, i] = c_opt
                
    return c_policy
