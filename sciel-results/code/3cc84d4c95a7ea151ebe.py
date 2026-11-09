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
    
    # Precompute derivative approximations for V_next
    # We will update this inside the loop
    
    # Define anchor points for the ratio search
    # We use a subset of the grid to solve for x, then interpolate
    # Using ~20 anchors should be fast and accurate
    anchor_indices = np.linspace(0, n_m - 1, 20, dtype=int)
    anchor_m = resource_grid[anchor_indices]
    
    for t in range(horizon - 2, -1, -1):
        V_next = V[t + 1]  # shape (n_z, n_m)
        
        # Precompute derivative approximations for V_next
        V_next_deriv = np.zeros_like(V_next)
        for z in range(n_z):
            # Central difference for interior points
            V_next_deriv[z, 1:-1] = (V_next[z, 2:] - V_next[z, :-2]) / (resource_grid[2:] - resource_grid[:-2])
            # One-sided at boundaries
            V_next_deriv[z, 0] = (V_next[z, 1] - V_next[z, 0]) / (resource_grid[1] - resource_grid[0])
            V_next_deriv[z, -1] = (V_next[z, -1] - V_next[z, -2]) / (resource_grid[-1] - resource_grid[-2])
        
        for z in range(n_z):
            probs = transition[z]
            
            # Solve for x (consumption ratio) at anchor points
            x_anchors = np.zeros(len(anchor_m))
            
            for k, m_a in enumerate(anchor_m):
                # Define FOC in terms of x
                # u'(x * m_a) - beta * R * sum P * V'_{next}(R(1-x)m_a + y) = 0
                
                def foc_x(x):
                    c_val = x * m_a
                    if c_val <= 1e-12:
                        return np.inf # u' is huge
                    if x >= 1.0:
                        # If x > 1, m' < y. We restrict x to [0, 1]
                        pass
                    
                    m_prime_0 = R * (1.0 - x) * m_a + income[0]
                    m_prime_1 = R * (1.0 - x) * m_a + income[1]
                    
                    # Clip to grid range
                    m_prime_0_clipped = np.clip(m_prime_0, resource_grid[0], resource_grid[-1])
                    m_prime_1_clipped = np.clip(m_prime_1, resource_grid[0], resource_grid[-1])
                    
                    v_deriv_0 = np.interp(m_prime_0_clipped, resource_grid, V_next_deriv[0])
                    v_deriv_1 = np.interp(m_prime_1_clipped, resource_grid, V_next_deriv[1])
                    
                    cont_deriv = probs[0] * v_deriv_0 + probs[1] * v_deriv_1
                    
                    return utility_deriv(c_val) - beta * R * cont_deriv
                
                # Solve for x in [1e-6, 1.0]
                # We use a small epsilon for lower bound to avoid log(0) issues in derivative
                lower_x = 1e-6
                upper_x = 1.0
                
                f_lower = foc_x(lower_x)
                f_upper = foc_x(upper_x)
                
                if f_lower * f_upper > 0:
                    # No sign change
                    if f_lower < 0:
                        x_opt = lower_x
                    else:
                        x_opt = upper_x
                else:
                    try:
                        x_opt = brentq(foc_x, lower_x, upper_x, xtol=1e-10, rtol=1e-10)
                    except:
                        # Fallback: bisection
                        a, b = lower_x, upper_x
                        for _ in range(50):
                            mid = (a + b) / 2
                            if foc_x(a) * foc_x(mid) <= 0:
                                b = mid
                            else:
                                a = mid
                        x_opt = (a + b) / 2
                
                x_anchors[k] = x_opt
            
            # Interpolate x to the full grid
            # x is a function of m. Since m is geomspace, log(m) is linear.
            # Interpolating in log-space is often better for multiplicative scales.
            log_m_anchors = np.log(anchor_m)
            log_m_grid = np.log(resource_grid)
            
            x_grid = np.interp(log_m_grid, log_m_anchors, x_anchors)
            
            # Ensure x is within [0, 1]
            x_grid = np.clip(x_grid, 0.0, 1.0)
            
            # Calculate policy
            policy[t, z, :] = x_grid * resource_grid
            
            # Calculate Value
            c_vals = policy[t, z, :]
            m_prime_0 = R * (resource_grid - c_vals) + income[0]
            m_prime_1 = R * (resource_grid - c_vals) + income[1]
            
            m_prime_0_clipped = np.clip(m_prime_0, resource_grid[0], resource_grid[-1])
            m_prime_1_clipped = np.clip(m_prime_1, resource_grid[0], resource_grid[-1])
            
            v_0 = np.interp(m_prime_0_clipped, resource_grid, V_next[0])
            v_1 = np.interp(m_prime_1_clipped, resource_grid, V_next[1])
            
            cont_value = probs[0] * v_0 + probs[1] * v_1
            V[t, z, :] = utility(c_vals) + beta * cont_value
    
    return policy
