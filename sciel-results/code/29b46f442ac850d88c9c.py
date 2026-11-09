import numpy as np
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_m = len(resource_grid)
    n_z = len(income)
    
    policy = np.zeros((horizon, n_z, n_m))
    V = np.zeros((horizon, n_z, n_m))
    
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1.0 - gamma)) / (1.0 - gamma)
        
    def utility_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # Last period: consume all
    for z in range(n_z):
        policy[horizon - 1, z, :] = resource_grid
        V[horizon - 1, z, :] = utility(resource_grid)
    
    for t in range(horizon - 2, -1, -1):
        V_next = V[t + 1]  # shape (n_z, n_m)
        
        # Precompute derivative approximations for V_next using central differences
        V_next_deriv = np.zeros_like(V_next)
        for z in range(n_z):
            V_next_deriv[z, 1:-1] = (V_next[z, 2:] - V_next[z, :-2]) / (resource_grid[2:] - resource_grid[:-2])
            V_next_deriv[z, 0] = (V_next[z, 1] - V_next[z, 0]) / (resource_grid[1] - resource_grid[0])
            V_next_deriv[z, -1] = (V_next[z, -1] - V_next[z, -2]) / (resource_grid[-1] - resource_grid[-2])
        
        for z in range(n_z):
            probs = transition[z]
            
            # Vectorized FOC solving for consumption ratio alpha = c/m
            # FOC: u'(alpha*m) = beta * R * sum_z' P(z'|z) * V'_{t+1}(R*m*(1-alpha) + y_{z'}, z')
            # For CRRA: u'(c) = c^(-gamma), so u'(alpha*m) = (alpha*m)^(-gamma) = alpha^(-gamma) * m^(-gamma)
            # RHS: beta * R * sum P * V'(R*m*(1-alpha) + y)
            # For large m, V' ~ m^(-gamma) (homothetic), so both sides scale as m^(-gamma)
            # This means alpha is approximately independent of m for large m.
            # We solve for alpha in (0, 1) using the FOC.
            
            # Define the FOC in terms of alpha:
            # f(alpha) = u'(alpha * m) - beta * R * [probs[0]*V'_0(R*(m - alpha*m) + y0) + probs[1]*V'_1(R*(m - alpha*m) + y1)]
            # = (alpha*m)^(-gamma) - beta*R*[...]
            
            # We'll solve this for each m. Since the problem is nearly homothetic,
            # we can use the solution at one m as an initial guess for nearby m values.
            
            alphas = np.zeros(n_m)
            
            # Solve for alpha at each m, using previous solution as hint
            prev_alpha = 0.5
            for i_m in range(n_m):
                m = resource_grid[i_m]
                
                def foc_alpha(alpha):
                    c = alpha * m
                    m_prime_0 = R * (m - c) + income[0]
                    m_prime_1 = R * (m - c) + income[1]
                    
                    m_prime_0_clipped = np.clip(m_prime_0, resource_grid[0], resource_grid[-1])
                    m_prime_1_clipped = np.clip(m_prime_1, resource_grid[0], resource_grid[-1])
                    
                    v_deriv_0 = np.interp(m_prime_0_clipped, resource_grid, V_next_deriv[0])
                    v_deriv_1 = np.interp(m_prime_1_clipped, resource_grid, V_next_deriv[1])
                    
                    cont_deriv = probs[0] * v_deriv_0 + probs[1] * v_deriv_1
                    
                    return utility_deriv(c) - beta * R * cont_deriv
                
                lower = 1e-10
                upper = 1.0 - 1e-10
                
                # Use prev_alpha as a starting hint to narrow the bracket
                # Try a small window around prev_alpha first
                window = 0.05
                lo = max(lower, prev_alpha - window)
                hi = min(upper, prev_alpha + window)
                
                f_lo = foc_alpha(lo)
                f_hi = foc_alpha(hi)
                
                if f_lo * f_hi <= 0:
                    try:
                        alpha_opt = brentq(foc_alpha, lo, hi, xtol=1e-12, rtol=1e-10)
                    except:
                        # Expand to full range
                        try:
                            alpha_opt = brentq(foc_alpha, lower, upper, xtol=1e-12, rtol=1e-10)
                        except:
                            # Fallback: pick boundary
                            if f_lo < 0:
                                alpha_opt = lo
                            else:
                                alpha_opt = hi
                else:
                    # No sign change in window, try full range
                    f_lower_full = foc_alpha(lower)
                    f_upper_full = foc_alpha(upper)
                    
                    if f_lower_full * f_upper_full > 0:
                        # No sign change at all - use boundary
                        if f_lower_full < 0:
                            alpha_opt = lower
                        else:
                            alpha_opt = upper
                    else:
                        try:
                            alpha_opt = brentq(foc_alpha, lower, upper, xtol=1e-12, rtol=1e-10)
                        except:
                            alpha_opt = prev_alpha
                
                # Ensure valid
                if not np.isfinite(alpha_opt) or alpha_opt < lower or alpha_opt > upper:
                    alpha_opt = prev_alpha
                
                alphas[i_m] = alpha_opt
                prev_alpha = alpha_opt
            
            # Compute policy and values
            c_vals = alphas * resource_grid
            policy[t, z, :] = c_vals
            
            # Compute V[t, z, :]
            m_prime_0 = R * (resource_grid - c_vals) + income[0]
            m_prime_1 = R * (resource_grid - c_vals) + income[1]
            m_prime_0_clipped = np.clip(m_prime_0, resource_grid[0], resource_grid[-1])
            m_prime_1_clipped = np.clip(m_prime_1, resource_grid[0], resource_grid[-1])
            
            v_0 = np.interp(m_prime_0_clipped, resource_grid, V_next[0])
            v_1 = np.interp(m_prime_1_clipped, resource_grid, V_next[1])
            cont_value = probs[0] * v_0 + probs[1] * v_1
            
            V[t, z, :] = utility(c_vals) + beta * cont_value
    
    return policy
