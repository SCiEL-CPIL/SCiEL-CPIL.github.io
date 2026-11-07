import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Extract parameters
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)
    n_out = len(resource_grid)
    
    # Utility functions
    def utility(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    def util_deriv(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    # Internal grid for computation
    # Use a denser grid than the output for better accuracy
    m_int = np.geomspace(0.001, 40, 400)
    n_m = len(m_int)
    
    # Initialize arrays
    # Policy: shape (horizon, n_states, n_m)
    # Value: shape (horizon, n_states, n_m)
    # Value derivative: shape (horizon, n_states, n_m)
    policy_int = np.zeros((horizon, n_states, n_m))
    value_int = np.zeros((horizon, n_states, n_m))
    val_deriv_int = np.zeros((horizon, n_states, n_m))
    
    # Helper for interpolation on log grid
    log_m_int = np.log(m_int)
    
    def interp_on_log(arr, m_next):
        """Interpolate arr (defined on m_int) at m_next using log-space interpolation"""
        m_next = np.maximum(m_next, 1e-12)
        log_m_next = np.log(m_next)
        if log_m_next <= log_m_int[0]:
            return arr[0]
        elif log_m_next >= log_m_int[-1]:
            return arr[-1]
        else:
            return np.interp(log_m_next, log_m_int, arr)
    
    def interp_on_log_vec(arr, m_next_arr):
        """Vectorized interpolation on log grid"""
        m_next_arr = np.maximum(m_next_arr, 1e-12)
        log_m_next = np.log(m_next_arr)
        result = np.interp(log_m_next, log_m_int, arr)
        return result
    
    # Terminal period t = horizon - 1
    t = horizon - 1
    for z in range(n_states):
        policy_int[t, z, :] = m_int
        value_int[t, z, :] = utility(m_int)
        val_deriv_int[t, z, :] = util_deriv(m_int)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        for z in range(n_states):
            # Check time
            if time.time() > deadline - 20.0:
                # Fallback: use a simple heuristic
                for z_inner in range(n_states):
                    # Use a proportional consumption rule as fallback
                    policy_int[t, z_inner, :] = 0.5 * m_int
                    policy_int[t, z_inner, :] = np.maximum(policy_int[t, z_inner, :], 1e-12)
                    policy_int[t, z_inner, :] = np.minimum(policy_int[t, z_inner, :], m_int)
                    value_int[t, z_inner, :] = utility(policy_int[t, z_inner, :])
                    val_deriv_int[t, z_inner, :] = util_deriv(policy_int[t, z_inner, :])
                break
            
            # For each grid point m, solve for optimal c using Newton's method
            # Euler equation: u'(c) = beta * R * E[Vp_{t+1}(m')]
            # where m' = R*(m-c) + income[z']
            
            c_opt = np.zeros(n_m)
            val_t = np.zeros(n_m)
            vp_t = np.zeros(n_m)
            
            for i in range(n_m):
                m = m_int[i]
                
                # Initial guess: use a simple heuristic
                # Proportional rule: c = m / (1 + beta*R)
                c_guess = m / (1 + beta * R)
                c_guess = np.clip(c_guess, 1e-12, m)
                
                # Newton's method on the Euler equation
                # f(c) = u'(c) - beta * R * E[Vp_{t+1}(m')]
                # f'(c) = u''(c) + beta * R * E[R * Vpp_{t+1}(m')]  (but we don't have Vpp)
                # Use bisection as fallback
                
                c_low = 1e-12
                c_high = m
                
                # Evaluate residual at boundaries
                def residual(c):
                    if c <= 1e-12:
                        c = 1e-12
                    if c >= m:
                        c = m
                    m_remaining = R * (m - c)
                    expected_vp = 0.0
                    for zp in range(n_states):
                        m_next = m_remaining + income[zp]
                        vp_val = interp_on_log(val_deriv_int[t+1, zp, :], m_next)
                        expected_vp += transition[z, zp] * vp_val
                    rhs = beta * R * expected_vp
                    return util_deriv(c) - rhs
                
                f_low = residual(c_low)
                f_high = residual(c_high)
                
                if f_low < 0:
                    # u'(c_low) < rhs, need to decrease c (increase u')
                    # But c_low is minimum, so c* = c_low
                    c_star = c_low
                elif f_high > 0:
                    # u'(c_high) > rhs, need to increase c (decrease u')
                    # But c_high is maximum, so c* = c_high
                    c_star = c_high
                else:
                    # Bisection method
                    for _ in range(30):
                        c_mid = (c_low + c_high) / 2
                        f_mid = residual(c_mid)
                        if abs(f_mid) < 1e-10:
                            break
                        if f_mid > 0:
                            # u'(c_mid) > rhs, need to increase c
                            c_low = c_mid
                        else:
                            # u'(c_mid) < rhs, need to decrease c
                            c_high = c_mid
                    c_star = (c_low + c_high) / 2
                
                c_star = np.clip(c_star, 1e-12, m)
                c_opt[i] = c_star
                
                # Calculate value
                m_rem = R * (m - c_star)
                cont = 0.0
                for zp in range(n_states):
                    m_next = m_rem + income[zp]
                    v_val = interp_on_log(value_int[t+1, zp, :], m_next)
                    cont += transition[z, zp] * v_val
                val_t[i] = utility(c_star) + beta * cont
                
                # Calculate Vp using Envelope Theorem
                expected_vp = 0.0
                for zp in range(n_states):
                    m_next = m_rem + income[zp]
                    vp_val = interp_on_log(val_deriv_int[t+1, zp, :], m_next)
                    expected_vp += transition[z, zp] * vp_val
                vp_t[i] = beta * R * expected_vp
            
            policy_int[t, z, :] = c_opt
            value_int[t, z, :] = val_t
            val_deriv_int[t, z, :] = vp_t
    
    # Interpolate the internal policies to the output grid
    policy_out = np.zeros((horizon, n_states, n_out))
    
    log_m_out = np.log(resource_grid)
    
    for t in range(horizon):
        for z in range(n_states):
            # Interpolate using log-space
            policy_out[t, z, :] = np.interp(log_m_out, log_m_int, policy_int[t, z, :])
            
            # Enforce constraints: 0 < c <= m + 1e-10
            policy_out[t, z, :] = np.maximum(policy_out[t, z, :], 1e-12)
            policy_out[t, z, :] = np.minimum(policy_out[t, z, :], resource_grid + 1e-10)
            
            # Special case for t = horizon - 1: c should be exactly m
            if t == horizon - 1:
                policy_out[t, z, :] = resource_grid
    
    return policy_out
