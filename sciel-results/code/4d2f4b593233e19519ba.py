import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_m_target = len(resource_grid)
    n_z = len(income)
    
    # Use a finer internal grid for higher accuracy
    # 500 points is a good balance between accuracy and speed
    n_m_internal = 500
    internal_grid = np.geomspace(resource_grid[0], resource_grid[-1], n_m_internal)
    
    # Policy and value arrays on the internal grid
    policy_internal = np.zeros((horizon, n_z, n_m_internal))
    V_internal = np.zeros((horizon, n_z, n_m_internal))
    
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
        policy_internal[horizon - 1, z, :] = internal_grid
        V_internal[horizon - 1, z, :] = utility(internal_grid)
    
    # Precompute interpolators for the last period? No, we need them for t=horizon-2.
    # We will build them on the fly for each t.
    
    for t in range(horizon - 2, -1, -1):
        # Build interpolators for V[t+1]
        interpolators = []
        for z in range(n_z):
            # PCHIP is monotonicity-preserving and C1 continuous
            # V is increasing in m, so this is suitable
            interp = PchipInterpolator(internal_grid, V_internal[t + 1, z, :])
            interpolators.append(interp)
            
        for z in range(n_z):
            probs = transition[z]
            
            for i_m, m in enumerate(internal_grid):
                # Define the FOC
                def foc(c):
                    m_prime_0 = R * (m - c) + income[0]
                    m_prime_1 = R * (m - c) + income[1]
                    
                    # Clip to internal grid range
                    m_prime_0_clipped = np.clip(m_prime_0, internal_grid[0], internal_grid[-1])
                    m_prime_1_clipped = np.clip(m_prime_1, internal_grid[0], internal_grid[-1])
                    
                    # Get derivative from interpolator
                    v_deriv_0 = interpolators[0].deriv(1)(m_prime_0_clipped)
                    v_deriv_1 = interpolators[1].deriv(1)(m_prime_1_clipped)
                    
                    cont_deriv = probs[0] * v_deriv_0 + probs[1] * v_deriv_1
                    
                    return utility_deriv(c) - beta * R * cont_deriv
                
                lower = 1e-12
                upper = m
                
                if upper - lower < 1e-15:
                    c_opt = upper
                else:
                    f_lower = foc(lower)
                    f_upper = foc(upper)
                    
                    if f_lower * f_upper > 0:
                        if f_lower < 0:
                            c_opt = lower
                        else:
                            c_opt = upper
                    else:
                        try:
                            c_opt = brentq(foc, lower, upper, xtol=1e-12, rtol=1e-10)
                        except:
                            # Fallback to bisection if brentq fails
                            try:
                                from scipy.optimize import bisect
                                c_opt = bisect(foc, lower, upper, xtol=1e-12)
                            except:
                                c_opt = upper # Conservative fallback
                    
                    if not np.isfinite(c_opt) or c_opt < lower or c_opt > upper:
                        c_opt = upper
                
                policy_internal[t, z, i_m] = c_opt
                
                # Compute V[t, z, i_m]
                m_prime_0 = R * (m - c_opt) + income[0]
                m_prime_1 = R * (m - c_opt) + income[1]
                m_prime_0_clipped = np.clip(m_prime_0, internal_grid[0], internal_grid[-1])
                m_prime_1_clipped = np.clip(m_prime_1, internal_grid[0], internal_grid[-1])
                
                v_0 = interpolators[0](m_prime_0_clipped)
                v_1 = interpolators[1](m_prime_1_clipped)
                cont_value = probs[0] * v_0 + probs[1] * v_1
                V_internal[t, z, i_m] = utility(c_opt) + beta * cont_value
    
    # Interpolate the policy back to the target resource_grid
    policy = np.zeros((horizon, n_z, n_m_target))
    for z in range(n_z):
        for t in range(horizon):
            policy[t, z, :] = np.interp(resource_grid, internal_grid, policy_internal[t, z, :])
            
    return policy
