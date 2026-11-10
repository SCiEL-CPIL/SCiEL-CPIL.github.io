import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
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
    
    # Use a dense working grid for high accuracy
    n_work = 2001
    work_grid = np.geomspace(0.001, 40, n_work)
    m_min = work_grid[0]
    m_max = work_grid[-1]
    
    # Utility and derivatives
    if gamma == 1:
        def utility(c):
            return np.log(np.maximum(c, c_floor))
        def u_prime(c):
            return 1.0 / np.maximum(c, c_floor)
        def u_double_prime(c):
            return -1.0 / (np.maximum(c, c_floor) ** 2)
    else:
        def utility(c):
            return np.power(np.maximum(c, c_floor), 1.0 - gamma) / (1.0 - gamma)
        def u_prime(c):
            return np.power(np.maximum(c, c_floor), -gamma)
        def u_double_prime(c):
            return -gamma * np.power(np.maximum(c, c_floor), -gamma - 1.0)
    
    policy_work = np.zeros((horizon, num_states, n_work))
    value_work = np.zeros((horizon, num_states, n_work))
    
    # Terminal condition: consume all
    t = horizon - 1
    for z in range(num_states):
        value_work[t, z, :] = utility(work_grid)
        policy_work[t, z, :] = work_grid
    
    # Build PCHIP interpolators for value and value derivative at terminal period
    v_interp = [None] * num_states
    vp_interp = [None] * num_states
    for zp in range(num_states):
        v_interp[zp] = PchipInterpolator(work_grid, value_work[t, zp, :])
        vp_interp[zp] = PchipInterpolator(work_grid, u_prime(policy_work[t, zp, :]))
    
    # Backward induction: solve Euler FOC via Newton-bisection hybrid
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break
        
        for z in range(num_states):
            probs = transition[z]
            nonzero_zp = [zp for zp in range(num_states) if probs[zp] > 1e-15]
            
            # Initial guess using heuristic
            m = work_grid
            if gamma == 1:
                alpha = 0.25 / (m * 0.3 + 0.5)
                alpha = np.clip(alpha, 0.02, 0.45)
            else:
                alpha = 0.3 / (m * 0.2 + 0.8)
                alpha = np.clip(alpha, 0.03, 0.5)
            c = m * (1.0 - alpha)
            c = np.clip(c, c_floor, m)
            
            c_low = np.full(n_work, c_floor)
            c_high = m.copy()
            
            max_iter = 80
            tol = 1e-13
            converged = False
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute expected next-period marginal utility using PCHIP
                rhs = np.zeros(n_work)
                for zp in nonzero_zp:
                    m_next = R * (m - c) + income[zp]
                    m_next_clipped = np.clip(m_next, m_min, m_max)
                    vprime_vals = vp_interp[zp](m_next_clipped)
                    rhs += probs[zp] * vprime_vals
                
                rhs = beta * R * rhs
                lhs = u_prime(c)
                
                f_val = lhs - rhs
                
                # f is decreasing in c
                too_low = f_val > 0
                c_low = np.where(too_low, c, c_low)
                c_high = np.where(~too_low, c, c_high)
                
                if np.all(c_high - c_low < tol):
                    converged = True
                    break
                
                # Newton step with second-order correction
                dc = np.maximum(np.abs(c) * 1e-5, 1e-11)
                c_plus = np.clip(c + dc, c_floor, m)
                
                rhs_plus = np.zeros(n_work)
                for zp in nonzero_zp:
                    m_next_plus = R * (m - c_plus) + income[zp]
                    m_next_plus_clipped = np.clip(m_next_plus, m_min, m_max)
                    vprime_vals_plus = vp_interp[zp](m_next_plus_clipped)
                    rhs_plus += probs[zp] * vprime_vals_plus
                rhs_plus = beta * R * rhs_plus
                
                f_plus = u_prime(c_plus) - rhs_plus
                df_dc = (f_plus - f_val) / dc
                
                # Second derivative estimate
                dc2 = np.maximum(np.abs(c) * 2e-5, 1e-10)
                c_minus = np.clip(c - dc2, c_floor, m)
                rhs_minus = np.zeros(n_work)
                for zp in nonzero_zp:
                    m_next_minus = R * (m - c_minus) + income[zp]
                    m_next_minus_clipped = np.clip(m_next_minus, m_min, m_max)
                    vprime_vals_minus = vp_interp[zp](m_next_minus_clipped)
                    rhs_minus += probs[zp] * vprime_vals_minus
                rhs_minus = beta * R * rhs_minus
                f_minus = u_prime(c_minus) - rhs_minus
                
                d2f_dc2 = (f_plus - 2*f_val + f_minus) / (dc * dc2)
                
                # Newton step with safeguard
                newton_c = c - f_val / np.where(np.abs(df_dc) > 1e-15, df_dc, 1e-15)
                newton_c = np.clip(newton_c, c_floor, m)
                
                bisection_mid = 0.5 * (c_low + c_high)
                
                use_newton = (newton_c > c_low) & (newton_c < c_high)
                c = np.where(use_newton, newton_c, bisection_mid)
                c = np.clip(c, c_floor, m)
            
            policy_work[t, z, :] = c
            
            # Compute value function using PCHIP
            u_val = utility(c)
            cont_val = np.zeros(n_work)
            for zp in nonzero_zp:
                m_next = R * (m - c) + income[zp]
                m_next_clipped = np.clip(m_next, m_min, m_max)
                v_next = v_interp[zp](m_next_clipped)
                cont_val += probs[zp] * v_next
            value_work[t, z, :] = u_val + beta * cont_val
        
        # Update PCHIP interpolators for this period
        for zp in range(num_states):
            v_interp[zp] = PchipInterpolator(work_grid, value_work[t, zp, :])
            vp_interp[zp] = PchipInterpolator(work_grid, u_prime(policy_work[t, zp, :]))
    
    # Interpolate from work grid to original grid using PCHIP
    policy = np.zeros((horizon, num_states, num_resources))
    for t in range(horizon):
        for z in range(num_states):
            p_interp = PchipInterpolator(work_grid, policy_work[t, z, :])
            policy[t, z, :] = p_interp(orig_grid)
            policy[t, z, :] = np.clip(policy[t, z, :], c_floor, orig_grid)
    
    # Terminal period must equal grid exactly
    for z in range(num_states):
        policy[horizon-1, z, :] = orig_grid
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_states):
            policy[t, z, :] = np.maximum(policy[t, z, :], c_floor)
            policy[t, z, :] = np.minimum(policy[t, z, :], orig_grid + 1e-10)
    
    return policy
