import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
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
    
    # Use a very dense working grid for high accuracy
    n_work = 801
    work_grid = np.geomspace(0.001, 40, n_work)
    
    # Utility and marginal utility
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
            return -gamma * np.power(np.maximum(c, c_floor), -gamma - 1)
    
    policy_work = np.zeros((horizon, num_states, n_work))
    value_work = np.zeros((horizon, num_states, n_work))
    
    # Terminal condition: consume all
    t = horizon - 1
    for z in range(num_states):
        value_work[t, z, :] = utility(work_grid)
        policy_work[t, z, :] = work_grid
    
    # Backward induction: solve Euler FOC via Newton-bisection hybrid
    # FOC: u'(c) = beta * R * E_z'[ V'_{t+1}(z', R*(m-c) + y_{z'}) ]
    # where V'_{t+1}(z', m) = u'(c*(z', m)) at the optimal policy
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break
        
        # Precompute V'_{t+1} on work_grid for each z'
        # V'_{t+1}(z', m) = u'(c*(z', m))
        vp_next = np.empty((num_states, n_work))
        for zp in range(num_states):
            vp_next[zp, :] = u_prime(policy_work[t+1, zp, :])
        
        for z in range(num_states):
            probs = transition[z]
            nonzero_zp = [zp for zp in range(num_states) if probs[zp] > 1e-15]
            
            # Initial guess: use a heuristic saving rule
            # For CRRA with log (gamma=1), the optimal saving behavior depends on m
            # Low m: save more (precautionary motive), High m: save less
            m = work_grid
            # Heuristic: c ~ m * (1 - alpha) where alpha depends on m and parameters
            # alpha ~ k / (m + b) for some k, b
            # Use a smooth approximation
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
            tol = 1e-12
            converged = False
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute expected next-period marginal utility
                rhs = np.zeros(n_work)
                for zp in nonzero_zp:
                    m_next = R * (m - c) + income[zp]
                    m_next_clipped = np.clip(m_next, work_grid[0], work_grid[-1])
                    vprime_vals = np.interp(m_next_clipped, work_grid, vp_next[zp, :])
                    rhs += probs[zp] * vprime_vals
                
                rhs = beta * R * rhs
                lhs = u_prime(c)
                
                f_val = lhs - rhs  # f(c) = u'(c) - beta*R*E[V'(m_next)]
                
                # f is decreasing in c (strictly), so:
                # f > 0 => c too low, need to increase c
                # f < 0 => c too high, need to decrease c
                too_low = f_val > 0
                c_low = np.where(too_low, c, c_low)
                c_high = np.where(~too_low, c, c_high)
                
                if np.all(c_high - c_low < tol):
                    converged = True
                    break
                
                # Newton step with numerical second derivative estimate
                dc = np.maximum(np.abs(c) * 5e-6, 1e-11)
                c_plus = np.clip(c + dc, c_floor, m)
                
                rhs_plus = np.zeros(n_work)
                for zp in nonzero_zp:
                    m_next_plus = R * (m - c_plus) + income[zp]
                    m_next_plus_clipped = np.clip(m_next_plus, work_grid[0], work_grid[-1])
                    vprime_vals_plus = np.interp(m_next_plus_clipped, work_grid, vp_next[zp, :])
                    rhs_plus += probs[zp] * vprime_vals_plus
                rhs_plus = beta * R * rhs_plus
                
                f_plus = u_prime(c_plus) - rhs_plus
                df_dc = (f_plus - f_val) / dc
                
                # Safeguard: if Newton step goes outside bounds, use bisection
                newton_c = c - f_val / np.where(np.abs(df_dc) > 1e-15, df_dc, 1e-15)
                newton_c = np.clip(newton_c, c_floor, m)
                
                bisection_mid = 0.5 * (c_low + c_high)
                
                # Use Newton if it stays within the bracket, otherwise bisection
                use_newton = (newton_c > c_low) & (newton_c < c_high)
                c = np.where(use_newton, newton_c, bisection_mid)
                c = np.clip(c, c_floor, m)
            
            policy_work[t, z, :] = c
            
            # Compute value function
            u_val = utility(c)
            cont_val = np.zeros(n_work)
            for zp in nonzero_zp:
                m_next = R * (m - c) + income[zp]
                m_next_clipped = np.clip(m_next, work_grid[0], work_grid[-1])
                v_next = np.interp(m_next_clipped, work_grid, value_work[t+1, zp, :])
                cont_val += probs[zp] * v_next
            value_work[t, z, :] = u_val + beta * cont_val
    
    # Interpolate from work grid to original grid
    policy = np.zeros((horizon, num_states, num_resources))
    for t in range(horizon):
        for z in range(num_states):
            policy[t, z, :] = np.interp(orig_grid, work_grid, policy_work[t, z, :])
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
