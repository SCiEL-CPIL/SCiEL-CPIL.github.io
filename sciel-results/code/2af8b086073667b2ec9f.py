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
    
    # Policy and value arrays
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
    
    # For each period t (going backward), we'll solve for the consumption ratio theta = c/m
    # at each state z, then compute c = theta * m for all m in the grid.
    # The key insight: for CRRA utility, the problem has a homothetic structure.
    # We represent the continuation value function using interpolation on the grid.
    
    for t in range(horizon - 2, -1, -1):
        V_next = V[t + 1]  # shape (n_z, n_m)
        
        # Precompute derivative approximations for V_next using central differences
        V_next_deriv = np.zeros_like(V_next)
        for z in range(n_z):
            # Central difference for interior points
            V_next_deriv[z, 1:-1] = (V_next[z, 2:] - V_next[z, :-2]) / (resource_grid[2:] - resource_grid[:-2])
            # One-sided at boundaries
            V_next_deriv[z, 0] = (V_next[z, 1] - V_next[z, 0]) / (resource_grid[1] - resource_grid[0])
            V_next_deriv[z, -1] = (V_next[z, -1] - V_next[z, -2]) / (resource_grid[-1] - resource_grid[-2])
        
        for z in range(n_z):
            probs = transition[z]
            
            # For each m in the grid, solve for optimal c
            # Use the FOC: u'(c) = beta * R * sum_z' P(z'|z) * V'_{t+1}(R(m-c) + y_{z'}, z')
            # We search over c in (0, m)
            
            for i_m, m in enumerate(resource_grid):
                # Define the FOC function
                def foc(c):
                    m_prime_0 = R * (m - c) + income[0]
                    m_prime_1 = R * (m - c) + income[1]
                    
                    # Clip to grid range
                    m_prime_0_clipped = np.clip(m_prime_0, resource_grid[0], resource_grid[-1])
                    m_prime_1_clipped = np.clip(m_prime_1, resource_grid[0], resource_grid[-1])
                    
                    v_deriv_0 = np.interp(m_prime_0_clipped, resource_grid, V_next_deriv[0])
                    v_deriv_1 = np.interp(m_prime_1_clipped, resource_grid, V_next_deriv[1])
                    
                    cont_deriv = probs[0] * v_deriv_0 + probs[1] * v_deriv_1
                    
                    return utility_deriv(c) - beta * R * cont_deriv
                
                # Solve for c in (1e-12, m)
                lower = 1e-12
                upper = m
                
                if upper - lower < 1e-15:
                    c_opt = upper
                else:
                    # Check signs at boundaries
                    f_lower = foc(lower)
                    f_upper = foc(upper)
                    
                    if f_lower * f_upper > 0:
                        # No sign change - use boundary
                        if f_lower < 0:
                            c_opt = lower
                        else:
                            c_opt = upper
                    else:
                        try:
                            c_opt = brentq(foc, lower, upper, xtol=1e-12, rtol=1e-10)
                        except:
                            # Fallback: grid search then refine
                            c_grid = np.linspace(1e-12, m, 100)
                            foc_vals = np.array([foc(c) for c in c_grid])
                            # Find sign changes
                            sign_changes = np.where(np.sign(foc_vals[:-1]) * np.sign(foc_vals[1:]) < 0)[0]
                            if len(sign_changes) > 0:
                                # Use the first sign change
                                idx = sign_changes[0]
                                try:
                                    c_opt = brentq(foc, c_grid[idx], c_grid[idx+1], xtol=1e-12, rtol=1e-10)
                                except:
                                    c_opt = c_grid[idx]
                            else:
                                # Use the c with foc closest to 0
                                c_opt = c_grid[np.argmin(np.abs(foc_vals))]
                    
                    # Ensure c_opt is valid
                    if not np.isfinite(c_opt) or c_opt < lower or c_opt > upper:
                        c_opt = upper
                
                policy[t, z, i_m] = c_opt
                
                # Compute V[t, z, i_m]
                m_prime_0 = R * (m - c_opt) + income[0]
                m_prime_1 = R * (m - c_opt) + income[1]
                m_prime_0_clipped = np.clip(m_prime_0, resource_grid[0], resource_grid[-1])
                m_prime_1_clipped = np.clip(m_prime_1, resource_grid[0], resource_grid[-1])
                v_0 = np.interp(m_prime_0_clipped, resource_grid, V_next[0])
                v_1 = np.interp(m_prime_1_clipped, resource_grid, V_next[1])
                cont_value = probs[0] * v_0 + probs[1] * v_1
                V[t, z, i_m] = utility(c_opt) + beta * cont_value
    
    return policy
