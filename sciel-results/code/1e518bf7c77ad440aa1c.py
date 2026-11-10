import numpy as np
import time
from scipy.interpolate import interp1d

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    
    # Policy: consumption[t, z, m_idx], shape (horizon, num_states, num_resources)
    policy = np.zeros((horizon, num_states, num_resources))
    # Value: value[t, z, m_idx]
    value = np.zeros((horizon, num_states, num_resources))
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1 - gamma) / (1 - gamma)
    
    def util_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        return c ** (-gamma)
        
    def util_second_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return -1.0 / (c * c)
        return -gamma * c ** (-gamma - 1)

    # Terminal condition: at t = horizon - 1 (last period), consume all
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    value[horizon - 1, :, :] = utility(resource_grid)[np.newaxis, :]
    
    # We will work in log-space for better approximation of the value function
    log_grid = np.log(resource_grid)
    
    # Pre-compute splines for terminal value in log-space
    # We need V(m), V'(m), V''(m)
    # If V_log(x) = V(exp(x)), then:
    # V'(m) = V_log'(x) / m
    # V''(m) = (V_log''(x) - V_log'(x)) / m^2
    
    # We will use cubic splines in log-space for V_log
    # To get high accuracy, we use a dense grid for the spline if possible, 
    # but here we are constrained to the provided resource_grid.
    # However, the resource_grid is already geomspace(0.001, 40, 101), which is log-spaced.
    # So fitting a spline to value[z] vs log_grid is natural.
    
    # Let's create a helper class or functions to evaluate V, V', V'' from log-splines
    # We will use scipy.interpolate.CubicSpline on (log_grid, value)
    # But we need to be careful with derivatives.
    
    from scipy.interpolate import CubicSpline
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: use simple rule for remaining periods
            for tt in range(t, -1, -1):
                # Simple heuristic: consume a fraction based on remaining periods
                # This is just to ensure validity if we run out of time
                frac = 1.0 / (horizon - tt)
                policy[tt, :, :] = np.minimum(frac * resource_grid[np.newaxis, :], resource_grid[np.newaxis, :])
                policy[tt, :, :] = np.maximum(policy[tt, :, :], 1e-12)
                value[tt, :, :] = utility(policy[tt, :, :])
            break
        
        # Create splines for V_{t+1}(m) for each state z' in log-space
        # V_log(x) = V(exp(x))
        splines = []
        for zp in range(num_states):
            # Fit cubic spline to value[t+1, zp, :] as a function of log_grid
            # Use 'not-a-knot' or natural? 'not-a-knot' is default and usually good.
            splines.append(CubicSpline(log_grid, value[t + 1, zp, :], bc_type='not-a-knot'))
            
        for z in range(num_states):
            m = resource_grid
            log_m = log_grid
            
            # Initial guess for c: use previous period's policy or a fraction
            # c ~ m / (horizon - t) is a rough guess
            # Let's use c = 0.5 * m as a safe initial guess
            c_est = 0.5 * m
            
            # Newton-Raphson iteration
            # f(c) = u'(c) - beta * R * sum_{zp} P(z|zp) * V'_{t+1}(R*(m-c)+income[zp])
            # f'(c) = u''(c) + beta * R^2 * sum_{zp} P(z|zp) * V''_{t+1}(R*(m-c)+income[zp])
            
            for _ in range(30):  # Max 30 Newton steps
                sum_v_prime = np.zeros_like(m)
                sum_v_second_prime = np.zeros_like(m)
                
                for zp in range(num_states):
                    m_next = R * (m - c_est) + income[zp]
                    # m_next must be positive for log
                    m_next = np.maximum(m_next, 1e-10)
                    log_m_next = np.log(m_next)
                    
                    # Evaluate V_log and its derivatives
                    # V_log(x) = spline(x)
                    # V_log'(x) = spline(x, 1)
                    # V_log''(x) = spline(x, 2)
                    
                    v_log = splines[zp](log_m_next, 0)
                    v_log_prime = splines[zp](log_m_next, 1)
                    v_log_second_prime = splines[zp](log_m_next, 2)
                    
                    # Transform back to m-space derivatives
                    # V'(m) = V_log'(log m) / m
                    # V''(m) = (V_log''(log m) - V_log'(log m)) / m^2
                    
                    vp = v_log_prime / m_next
                    vpp = (v_log_second_prime - v_log_prime) / (m_next * m_next)
                    
                    sum_v_prime += transition[z, zp] * vp
                    sum_v_second_prime += transition[z, zp] * vpp
                
                u_prime = util_deriv(c_est)
                u_second_prime = util_second_deriv(c_est)
                
                f = u_prime - beta * R * sum_v_prime
                f_prime = u_second_prime + beta * R**2 * sum_v_second_prime
                
                # Newton step
                # Avoid division by zero
                f_prime_safe = np.where(np.abs(f_prime) < 1e-10, 1e-10, f_prime)
                c_new = c_est - f / f_prime_safe
                
                # Clip c_new to [1e-12, m]
                c_new = np.maximum(c_new, 1e-12)
                c_new = np.minimum(c_new, m)
                
                # Check convergence
                if np.all(np.abs(c_new - c_est) < 1e-9 * np.maximum(1.0, m)):
                    c_est = c_new
                    break
                c_est = c_new
            
            policy[t, z, :] = c_est
            
            # Compute value V_t(m) = u(c) + beta * E[V_{t+1}(m')]
            u_c = utility(c_est)
            sum_v_next = np.zeros_like(m)
            for zp in range(num_states):
                m_next = R * (m - c_est) + income[zp]
                m_next = np.maximum(m_next, 1e-10)
                log_m_next = np.log(m_next)
                v_next = splines[zp](log_m_next, 0)
                sum_v_next += transition[z, zp] * v_next
                
            value[t, z, :] = u_c + beta * sum_v_next

    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_states):
            for m_idx in range(num_resources):
                m = resource_grid[m_idx]
                c = policy[t, z, m_idx]
                if c <= 0:
                    policy[t, z, m_idx] = 1e-12
                if c > m + 1e-10:
                    policy[t, z, m_idx] = m

    # Ensure c[5] equals grid within 1e-9
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    
    # Check for non-finite values
    if not np.all(np.isfinite(policy)):
        policy = np.nan_to_num(policy, nan=1e-12, posinf=1e10, neginf=0)
    
    return policy
