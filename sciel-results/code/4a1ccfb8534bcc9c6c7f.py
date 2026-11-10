def solve_consumption(model, resource_grid, budget):
    import numpy as np
    from scipy.optimize import brentq
    from scipy.interpolate import make_interp_spline
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # resource_grid: 101 points from 0.001 to 40
    # We need to return consumption[t, income_state, resource_index] with shape (6, 2, 101)
    
    # Utility and marginal utility
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-12:
            return np.log(c)
        else:
            return c**(1 - gamma) / (1 - gamma)
    
    def up(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-12:
            return 1.0 / c
        else:
            return c**(-gamma)
    
    def up_inv(mv):
        # Invert marginal utility: c = up_inv(mv)
        mv = np.maximum(mv, 1e-12)
        if abs(gamma - 1) < 1e-12:
            return 1.0 / mv
        else:
            return mv**(-1.0 / gamma)
    
    # Use a fine grid for training the approximations
    # We'll use log-spaced grid
    n_train = 300
    m_train = np.geomspace(0.001, 40, n_train)
    log_m_train = np.log(m_train)
    
    # MV at terminal period: MV_5(m, z) = u'(m) for all z
    # We'll store MV as a function of (m, z). For each z, fit a smooth function.
    
    # At t=5, MV is independent of z, so we can just use up(m)
    # But we need to represent it as a smooth function for the next step.
    
    # Let's store the MV approximation as a polynomial in log(m) for each z.
    # Degree of polynomial
    poly_degree = 15
    
    # MV_t for each z: represented as coefficients of polynomial in log(m)
    # MV_t(z)(m) = sum_{k=0}^{deg} coeff[z, k] * (log(m))^k
    
    # At t=5: MV_5(m, z) = up(m) for all z
    # Fit up(m) as polynomial in log(m)
    log_m = np.log(m_train)
    mv5 = up(m_train)
    
    # Fit polynomial
    coeffs5 = np.polyfit(log_m, mv5, poly_degree)
    # Reverse to standard form (highest degree first from polyfit, we want lowest first)
    coeffs5 = coeffs5[::-1]
    
    # MV for each z at t=5 is the same
    MV_coeffs = np.zeros((2, poly_degree + 1))
    MV_coeffs[0] = coeffs5
    MV_coeffs[1] = coeffs5
    
    # Now go backward from t=4 to t=0
    # consumption[t, z, :] will be the consumption at period t, income state z, for each m in resource_grid
    consumption = np.zeros((horizon, 2, len(resource_grid)))
    
    # Terminal: t=5
    consumption[5, 0, :] = resource_grid
    consumption[5, 1, :] = resource_grid
    
    # For t = 4, 3, 2, 1, 0
    for t in range(4, -1, -1):
        # For each income state z, solve for optimal c(m, z)
        c_train = np.zeros((2, n_train))
        
        for z in range(2):
            for i in range(n_train):
                m = m_train[i]
                
                # Solve for c in: c = up_inv(beta * R * sum_{z'} P(z'|z) * MV_{t+1}(R(m-c) + income[z'], z'))
                def euler_eq(c_val):
                    if c_val <= 0 or c_val >= m:
                        return c_val - m * 0.999  # push away from boundaries
                    mv_next = 0.0
                    for zp in range(2):
                        m_next = R * (m - c_val) + income[zp]
                        if m_next < 1e-12:
                            m_next = 1e-12
                        log_m_next = np.log(m_next)
                        # Evaluate MV_{t+1}(m_next, zp) using polynomial
                        mv_zp = np.polyval(MV_coeffs[zp][::-1], log_m_next)
                        mv_next += transition[z, zp] * mv_zp
                    target_mv = beta * R * mv_next
                    c_target = up_inv(target_mv)
                    return c_val - c_target
                
                # Bounded search: c in (1e-10, m)
                lo = 1e-10
                hi = m - 1e-10
                if hi <= lo:
                    c_opt = m / 2
                else:
                    try:
                        c_opt = brentq(euler_eq, lo, hi, xtol=1e-14, rtol=1e-14)
                    except:
                        # Fallback: bisection or just use a heuristic
                        c_opt = m * 0.5
                
                c_train[z, i] = c_opt
            
            # Fit c_train[z, :] as polynomial in log(m)
            c_coeffs = np.polyfit(log_m, c_train[z, :], poly_degree)
            c_coeffs = c_coeffs[::-1]
            
            # Also compute MV_t(m, z) = up(c_train[z, :]) and fit it
            mv_train = up(c_train[z, :])
            mv_coeffs = np.polyfit(log_m, mv_train, poly_degree)
            mv_coeffs = mv_coeffs[::-1]
            
            MV_coeffs[z] = mv_coeffs
        
        # Now evaluate consumption on resource_grid for each z
        log_rg = np.log(resource_grid)
        for z in range(2):
            # Re-fit c on the training grid to get coefficients, then evaluate
            # We already have c_train[z, :], fit and evaluate
            c_coeffs = np.polyfit(log_m, c_train[z, :], poly_degree)
            c_coeffs_rev = c_coeffs[::-1]
            c_eval = np.polyval(c_coeffs_rev, log_rg)
            # Clip to valid range
            c_eval = np.clip(c_eval, 1e-12, resource_grid + 1e-10)
            consumption[t, z, :] = c_eval
    
    return consumption
