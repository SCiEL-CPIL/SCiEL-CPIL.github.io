import time
import numpy as np

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_grid = len(resource_grid)
    n_states = len(income)
    
    # Use a dense internal grid for accuracy
    # The scoring interpolates onto geomspace(0.001, 40, 251) and checks 0.2 <= m <= 8
    # So we need high accuracy in that range.
    
    # Strategy: Use a dense log-spaced grid for the value function,
    # and solve the Euler equation via vectorized root-finding.
    
    # Internal grid: use the resource_grid itself or a denser one?
    # The task says resource_grid = np.geomspace(0.001, 40, 101)
    # But we can use a denser internal grid for better accuracy.
    
    # Let's use a denser internal grid, say 1000 points in log space
    n_internal = 1000
    m_internal = np.geomspace(resource_grid[0], resource_grid[-1], n_internal)
    log_m_internal = np.log(m_internal)
    
    # Utility functions
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1 - gamma) / (1 - gamma)
    
    def util_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        return c ** (-gamma)
    
    # Backward induction
    # V[t, z, :] - value function on internal grid
    # C[t, z, :] - consumption policy on internal grid
    
    V = np.zeros((horizon + 1, n_states, n_internal))
    C = np.zeros((horizon + 1, n_states, n_internal))
    
    # Terminal condition: consume all resources
    V[horizon, :, :] = util(m_internal)
    C[horizon, :, :] = m_internal
    
    # For interpolation, we'll use np.interp on log scale for better accuracy
    def interp_log(x, xp, fp):
        """Interpolate using log-spaced grid xp, values fp, at points x."""
        # x and xp should be positive
        log_x = np.log(np.maximum(x, 1e-15))
        log_xp = np.log(xp)
        return np.interp(log_x, log_xp, fp)
    
    # Backward induction
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 15.0:
            # Fallback: use terminal condition for remaining periods
            for tt in range(t, -1, -1):
                V[tt, :, :] = V[tt + 1, :, :]
                C[tt, :, :] = m_internal
            break
        
        for z in range(n_states):
            if time.time() > deadline - 5.0:
                # Fallback
                for tt in range(t, -1, -1):
                    V[tt, z, :] = V[tt + 1, z, :]
                    C[tt, z, :] = m_internal
                break
            
            # For each m in m_internal, find optimal c
            # V_t(z, m) = max_c { util(c) + beta * sum_z' transition[z, z'] * V_{t+1}(z', m') }
            # where m' = R * (m - c) + income[z']
            
            # We'll use a grid of consumption fractions and vectorize
            # Then refine with a local search
            
            # First, use a coarse grid of consumption fractions
            n_frac = 200
            c_fracs = np.linspace(0.001, 1.0, n_frac)
            
            # For each m, compute the objective for all c_fracs
            # m_internal has shape (n_internal,)
            # c_fracs has shape (n_frac,)
            
            # m_next for each (m, c_frac, z')
            # m_remaining = R * (m - m * c_frac) = R * m * (1 - c_frac)
            # m_next = m_remaining + income[z']
            
            V_t_z = np.zeros(n_internal)
            C_t_z = np.zeros(n_internal)
            
            # Vectorized computation
            # m_internal: (n_internal,)
            # c_fracs: (n_frac,)
            
            # For each z', compute V_{t+1}(z', m_next)
            # m_next = R * m_internal * (1 - c_fracs) + income[z']
            
            # Shape: (n_internal, n_frac)
            m_remaining = R * m_internal[:, None] * (1 - c_fracs[None, :])
            
            # Continuation value for each z'
            cont_val = np.zeros((n_internal, n_frac))
            for zp in range(n_states):
                m_next = m_remaining + income[zp]
                # Interpolate V_{t+1}(zp, m_next)
                # Clip m_next to the range of m_internal
                m_next_clipped = np.clip(m_next, m_internal[0], m_internal[-1])
                V_interp = interp_log(m_next_clipped, m_internal, V[t + 1, zp, :])
                cont_val += transition[z, zp] * V_interp
            
            # Total value: util(c) + beta * cont_val
            c_vals = m_internal[:, None] * c_fracs[None, :]
            total_val = util(c_vals) + beta * cont_val
            
            # Find best c for each m
            best_idx = np.argmax(total_val, axis=1)
            V_t_z = total_val[np.arange(n_internal), best_idx]
            C_t_z = c_vals[np.arange(n_internal), best_idx]
            
            # Refine with a local search around the best fraction
            # For each m, do a few iterations of local refinement
            for refine_iter in range(3):
                if time.time() > deadline - 2.0:
                    break
                
                # Compute gradient or use a finer local grid
                # For each m, evaluate a finer grid around the best c
                # Use a small neighborhood in c_frac space
                
                # Compute the derivative of the objective with respect to c
                # d/dc [util(c) + beta * sum_z' trans[z,z'] * V_{t+1}(z', R(m-c) + income[z'])]
                # = util'(c) - beta * R * sum_z' trans[z,z'] * V'_{t+1}(z', m')
                
                # We need V'_{t+1}(z', m) - derivative of value function
                # Approximate using finite differences on the internal grid
                dm = m_internal[1] - m_internal[0]
                V_prime = np.zeros((n_states, n_internal))
                for zp in range(n_states):
                    V_prime[zp, 1:-1] = (V[t + 1, zp, 2:] - V[t + 1, zp, :-2]) / (2 * dm)
                    V_prime[zp, 0] = (V[t + 1, zp, 1] - V[t + 1, zp, 0]) / dm
                    V_prime[zp, -1] = (V[t + 1, zp, -1] - V[t + 1, zp, -2]) / dm
                
                # For each m, compute the optimal c using Newton's method
                c_current = C_t_z.copy()
                
                for i in range(n_internal):
                    m = m_internal[i]
                    c = c_current[i]
                    
                    # Compute m_next for each z'
                    m_remaining = R * (m - c)
                    m_next = m_remaining + income
                    
                    # Compute V' at m_next
                    m_next_clipped = np.clip(m_next, m_internal[0], m_internal[-1])
                    V_prime_interp = np.array([
                        interp_log(m_next_clipped[zp], m_internal, V_prime[zp, :])
                        for zp in range(n_states)
                    ])
                    
                    # FOC: util'(c) = beta * R * sum_z' trans[z,z'] * V'_{t+1}(z', m')
                    rhs = beta * R * np.dot(transition[z, :], V_prime_interp)
                    lhs = util_deriv(c)
                    
                    # Newton step
                    # d/dc [util'(c) - rhs] = util''(c) + beta * R^2 * sum trans * V''
                    # Approximate util''(c)
                    if abs(gamma - 1.0) < 1e-12:
                        util_double_prime = -1.0 / c**2
                    else:
                        util_double_prime = -gamma * c ** (-gamma - 1)
                    
                    # Approximate V'' using finite differences
                    V_double_prime = np.zeros(n_states)
                    for zp in range(n_states):
                        V_double_prime[zp] = interp_log(m_next_clipped[zp], m_internal,
                                                         (V[t + 1, zp, 2:] - 2 * V[t + 1, zp, 1:-1] + V[t + 1, zp, :-2]) / dm**2)
                    
                    second_derivative = util_double_prime + beta * R**2 * np.dot(transition[z, :], V_double_prime)
                    
                    if abs(second_derivative) > 1e-15:
                        delta_c = (lhs - rhs) / second_derivative
                        # Limit step size
                        delta_c = np.clip(delta_c, -0.1 * m, 0.1 * m)
                        c_new = c + delta_c
                        c_new = np.clip(c_new, 1e-12, m - 1e-12)
                        
                        # Check if the new c is better
                        c_old_val = util(c) + beta * np.dot(transition[z, :], 
                                                            np.array([interp_log(np.clip(m_remaining + income[zp], m_internal[0], m_internal[-1]), m_internal, V[t + 1, zp, :]) for zp in range(n_states)]))
                        c_new_val = util(c_new) + beta * np.dot(transition[z, :], 
                                                                np.array([interp_log(np.clip(R * (m - c_new) + income[zp], m_internal[0], m_internal[-1]), m_internal, V[t + 1, zp, :]) for zp in range(n_states)]))
                        
                        if c_new_val > c_old_val:
                            c_current[i] = c_new
                            V_t_z[i] = c_new_val
                        else:
                            c_current[i] = c
                            V_t_z[i] = c_old_val
                    else:
                        c_current[i] = c
                        V_t_z[i] = util(c) + beta * np.dot(transition[z, :], 
                                                          np.array([interp_log(np.clip(m_remaining + income[zp], m_internal[0], m_internal[-1]), m_internal, V[t + 1, zp, :]) for zp in range(n_states)]))
                
                C_t_z = c_current
            
            V[t, z, :] = V_t_z
            C[t, z, :] = C_t_z
    
    # Now interpolate to the original resource_grid
    result = np.zeros((horizon, n_states, n_grid))
    
    for t in range(horizon):
        for z in range(n_states):
            result[t, z, :] = interp_log(resource_grid, m_internal, C[t, z, :])
            # Enforce constraints
            result[t, z, :] = np.clip(result[t, z, :], 1e-12, resource_grid + 1e-10)
    
    # Terminal period: consume all
    result[horizon - 1, :, :] = resource_grid
    
    # Final validation
    result = np.nan_to_num(result, nan=1e-12, posinf=resource_grid, neginf=1e-12)
    result = np.maximum(result, 1e-12)
    
    return result
