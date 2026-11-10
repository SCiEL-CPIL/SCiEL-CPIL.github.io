import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'])
    transition = np.asarray(model['transition'])
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    
    # Utility and derivative functions
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    def utility_derivative(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    # Policy and Value arrays
    policy = np.zeros((horizon, num_states, num_resources))
    value = np.zeros((horizon, num_states, num_resources))
    
    # Terminal condition: consume all resources
    policy[horizon - 1, :, :] = resource_grid[None, None, :]
    value[horizon - 1, :, :] = utility(resource_grid)[None, None, :]
    
    # Backward induction for t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        for z in range(num_states):
            for m_idx in range(num_resources):
                m = resource_grid[m_idx]
                
                # Define the objective function and its derivative
                def objective(c):
                    c = np.clip(c, 1e-12, m)
                    cont_val = 0.0
                    for zp in range(num_states):
                        m_next = R * (m - c) + income[zp]
                        if m_next <= resource_grid[0]:
                            v_next = value[t + 1, zp, 0]
                        elif m_next >= resource_grid[-1]:
                            v_next = value[t + 1, zp, -1]
                        else:
                            v_next = np.interp(m_next, resource_grid, value[t + 1, zp, :])
                        cont_val += transition[z, zp] * v_next
                    return utility(c) + beta * cont_val
                
                def objective_derivative(c):
                    c = np.clip(c, 1e-12, m)
                    deriv_util = utility_derivative(c)
                    deriv_cont = 0.0
                    for zp in range(num_states):
                        m_next = R * (m - c) + income[zp]
                        # Derivative of interpolated value w.r.t. m_next
                        if m_next <= resource_grid[0]:
                            dv_dm = 0.0
                        elif m_next >= resource_grid[-1]:
                            dv_dm = 0.0
                        else:
                            # Find the segment
                            idx = np.searchsorted(resource_grid, m_next) - 1
                            idx = np.clip(idx, 0, num_resources - 2)
                            m_lo = resource_grid[idx]
                            m_hi = resource_grid[idx + 1]
                            v_lo = value[t + 1, zp, idx]
                            v_hi = value[t + 1, zp, idx + 1]
                            if m_hi - m_lo > 1e-15:
                                dv_dm = (v_hi - v_lo) / (m_hi - m_lo)
                            else:
                                dv_dm = 0.0
                        deriv_cont += transition[z, zp] * dv_dm
                    # d/dc of continuation: beta * sum transition[z,zp] * (-R) * dv_dm
                    return deriv_util - beta * R * deriv_cont
                
                # Use Newton's method with bisection fallback
                a = 1e-12
                b = m
                
                # First check if optimum is at boundary
                f_a = objective_derivative(a)
                f_b = objective_derivative(b)
                
                if f_a <= 0:
                    c_opt = a
                elif f_b >= 0:
                    c_opt = b
                else:
                    # Interior optimum: use Newton with bisection
                    c = (a + b) / 2.0
                    converged = False
                    
                    for _ in range(50):
                        if time.time() > deadline:
                            break
                        
                        f = objective_derivative(c)
                        
                        if abs(f) < 1e-10:
                            converged = True
                            break
                        
                        # Update bracket
                        if f > 0:
                            a = c
                        else:
                            b = c
                        
                        # Newton step
                        c_newton = c - f / objective_derivative(c)
                        
                        # Check if Newton step is in bracket
                        if a < c_newton < b:
                            c = c_newton
                        else:
                            c = (a + b) / 2.0
                        
                        if abs(b - a) < 1e-10:
                            converged = True
                            break
                    
                    c_opt = c
                
                c_opt = np.clip(c_opt, 1e-12, m)
                policy[t, z, m_idx] = c_opt
                value[t, z, m_idx] = objective(c_opt)
        
        if time.time() > deadline:
            break
    
    # Ensure terminal condition is correct
    policy[horizon - 1, :, :] = resource_grid[None, None, :]
    
    # Validate: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_states):
            for m_idx in range(num_resources):
                m = resource_grid[m_idx]
                c = policy[t, z, m_idx]
                if c <= 0:
                    policy[t, z, m_idx] = 1e-12
                if c > m + 1e-10:
                    policy[t, z, m_idx] = m
    
    return policy
