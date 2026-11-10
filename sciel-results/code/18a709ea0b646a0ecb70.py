import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 142  # Safety margin before 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'], dtype=np.float64)
    transition = np.array(model['transition'], dtype=np.float64)
    horizon = model['horizon']
    
    num_states = len(income)
    orig_grid = resource_grid
    num_resources = len(resource_grid)
    c_floor = 1e-12
    
    # Use a very dense internal grid for accurate interpolation
    n_work = 3001
    work_grid = np.geomspace(0.001, 40, n_work)
    
    # Precompute log-spaced grid for faster interpolation
    log_work_grid = np.log(work_grid)
    
    # Utility and marginal utility functions
    if gamma == 1:
        def utility(c):
            c = np.maximum(c, c_floor)
            return np.log(c)
        def u_prime(c):
            c = np.maximum(c, c_floor)
            return 1.0 / c
        def u_prime_log(c):
            c = np.maximum(c, c_floor)
            return -np.log(c)
    else:
        def utility(c):
            c = np.maximum(c, c_floor)
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)
        def u_prime(c):
            c = np.maximum(c, c_floor)
            return np.power(c, -gamma)
        def u_prime_log(c):
            c = np.maximum(c, c_floor)
            return -gamma * np.log(c)
    
    policy_work = np.zeros((horizon, num_states, n_work))
    value_work = np.zeros((horizon, num_states, n_work))
    
    # Terminal condition: consume all
    t = horizon - 1
    for z in range(num_states):
        value_work[t, z, :] = utility(work_grid)
        policy_work[t, z, :] = work_grid
    
    # Precompute nonzero transitions
    nonzero_transitions = []
    for z in range(num_states):
        probs = transition[z]
        nonzero_zp = [(zp, probs[zp]) for zp in range(num_states) if probs[zp] > 1e-15]
        nonzero_transitions.append(nonzero_zp)
    
    # Backward induction: solve Euler equation at each grid point
    # FOC: u'(c) = beta * R * sum_zp P(zp|z) * V'(R*(m-c) + income[zp])
    # This is a 1D root-finding problem for each m
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Return best effort so far
            break
        
        for z in range(num_states):
            nonzero_zp = nonzero_transitions[z]
            m = work_grid
            
            # Initial guess: use a smooth heuristic
            # For CRRA, optimal saving behavior is well-studied
            # Use a logistic-type saving rate
            saving_rate = 0.3 / (1.0 + 0.3 * m)
            c = np.clip(m * (1.0 - saving_rate), c_floor, m)
            
            # Bisection bounds
            c_low = np.full(n_work, c_floor)
            c_high = m.copy()
            
            max_iter = 60
            tol = 1e-14
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute RHS of FOC: beta * R * E[V'(m_next)]
                # V'(m_next) = u'(c_next(m_next))
                rhs = np.zeros(n_work)
                for zp, prob in nonzero_zp:
                    m_next = R * (m - c) + income[zp]
                    # Clip to work grid range
                    m_next_clipped = np.clip(m_next, work_grid[0], work_grid[-1])
                    # Interpolate V' from next period using log-space for better accuracy
                    vprime_vals = np.interp(m_next_clipped, work_grid, 
                                           u_prime(policy_work[t+1, zp, :]))
                    rhs += prob * vprime_vals
                rhs = beta * R * rhs
                
                # LHS = u'(c)
                lhs = u_prime(c)
                
                # f(c) = lhs - rhs
                # f is decreasing in c
                too_low = lhs > rhs
                c_low = np.where(too_low, c, c_low)
                c_high = np.where(~too_low, c, c_high)
                
                # Check convergence
                if np.all(c_high - c_low < tol):
                    break
                
                # Halley's method or Newton with numerical derivative
                dc = np.maximum(np.abs(c) * 1e-7, 1e-14)
                c_plus = np.clip(c + dc, c_floor, m)
                
                rhs_plus = np.zeros(n_work)
                for zp, prob in nonzero_zp:
                    m_next_plus = R * (m - c_plus) + income[zp]
                    m_next_plus_clipped = np.clip(m_next_plus, work_grid[0], work_grid[-1])
                    vprime_vals_plus = np.interp(m_next_plus_clipped, work_grid, 
                                               u_prime(policy_work[t+1, zp, :]))
                    rhs_plus += prob * vprime_vals_plus
                rhs_plus = beta * R * rhs_plus
                
                f_val = lhs - rhs
                f_plus = u_prime(c_plus) - rhs_plus
                df_dc = (f_plus - f_val) / dc
                
                # Second derivative for Halley's method
                dc2 = np.maximum(np.abs(c) * 1e-7, 1e-14)
                c_minus = np.clip(c - dc2, c_floor, m)
                rhs_minus = np.zeros(n_work)
                for zp, prob in nonzero_zp:
                    m_next_minus = R * (m - c_minus) + income[zp]
                    m_next_minus_clipped = np.clip(m_next_minus, work_grid[0], work_grid[-1])
                    vprime_vals_minus = np.interp(m_next_minus_clipped, work_grid, 
                                               u_prime(policy_work[t+1, zp, :]))
                    rhs_minus += prob * vprime_vals_minus
                rhs_minus = beta * R * rhs_minus
                
                f_minus = u_prime(c_minus) - rhs_minus
                df_dc_minus = (f_val - f_minus) / dc2
                d2f_dc2 = (df_dc - df_dc_minus) / (2 * dc2)
                
                # Halley's step
                safe_df = np.where(np.abs(df_dc) > 1e-16, df_dc, 1e-16)
                halley_denom = 1.0 - 0.5 * f_val * d2f_dc2 / safe_df
                halley_denom = np.where(np.abs(halley_denom) > 1e-12, halley_denom, 1.0)
                halley_c = c - f_val / (safe_df * halley_denom)
                
                # Use Halley if within bounds, else Newton, else bisection
                bisection_mid = (c_low + c_high) / 2.0
                newton_c = c - f_val / safe_df
                
                c = np.where(
                    (halley_c > c_low) & (halley_c < c_high),
                    halley_c,
                    np.where(
                        (newton_c > c_low) & (newton_c < c_high),
                        newton_c,
                        bisection_mid
                    )
                )
                c = np.clip(c, c_floor, m)
            
            policy_work[t, z, :] = c
            
            # Compute value
            u_val = utility(c)
            cont_val = np.zeros(n_work)
            for zp, prob in nonzero_zp:
                m_next = R * (m - c) + income[zp]
                m_next_clipped = np.clip(m_next, work_grid[0], work_grid[-1])
                v_next = np.interp(m_next_clipped, work_grid, value_work[t+1, zp, :])
                cont_val += prob * v_next
            value_work[t, z, :] = u_val + beta * cont_val
    
    # Interpolate from work grid to original grid
    policy = np.zeros((horizon, num_states, num_resources))
    for t in range(horizon):
        for z in range(num_states):
            policy[t, z, :] = np.interp(orig_grid, work_grid, policy_work[t, z, :])
            # Enforce bounds: 0 < c <= m
            policy[t, z, :] = np.clip(policy[t, z, :], c_floor, orig_grid)
    
    # Final validity checks
    # Terminal period must equal grid
    for z in range(num_states):
        policy[horizon-1, z, :] = orig_grid
        policy[horizon-1, z, :] = np.clip(policy[horizon-1, z, :], c_floor, orig_grid)
    
    # Ensure 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_states):
            policy[t, z, :] = np.maximum(policy[t, z, :], c_floor)
            policy[t, z, :] = np.minimum(policy[t, z, :], orig_grid + 1e-10)
    
    return policy
