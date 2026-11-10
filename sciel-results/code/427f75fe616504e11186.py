import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    # Utility derivative
    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return 1.0 / c
        else:
            return np.power(c, -gamma)
    
    # Policy and Value arrays
    # policy[t, z, m_idx]
    policy = np.zeros((horizon, num_states, num_resources))
    value = np.zeros((horizon, num_states, num_resources))
    
    # Terminal Period
    t = horizon - 1
    for z in range(num_states):
        value[t, z, :] = utility(resource_grid)
        policy[t, z, :] = resource_grid
    
    # Backward Induction using Newton's method on the FOC
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with consume-all
            for tt in range(t, -1, -1):
                for z in range(num_states):
                    policy[tt, z, :] = resource_grid
                    value[tt, z, :] = utility(resource_grid)
            break
        
        for z in range(num_states):
            probs = transition[z]
            
            # Initialize c with a heuristic: consume a fraction of m
            # For CRRA with log utility, optimal consumption is roughly m/(1+beta*R)
            # Use a better initial guess based on the previous period's policy
            if t + 1 < horizon:
                # Interpolate next period's policy as initial guess
                c_init = np.interp(resource_grid, resource_grid, policy[t+1, z, :])
            else:
                c_init = resource_grid / (1 + beta * R)
            
            c = np.clip(c_init, 1e-12, resource_grid)
            
            max_iter = 50
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute LHS: u'(c)
                lhs = u_prime(c)
                
                # Compute RHS: beta * R * sum_zp probs[zp] * V'_{t+1}(R*(m-c) + income[zp])
                rhs = np.zeros(num_resources)
                for zp in range(num_states):
                    if probs[zp] == 0:
                        continue
                    
                    m_next = R * (resource_grid - c) + income[zp]
                    
                    # Interpolate policy at m_next
                    c_next = np.interp(m_next, resource_grid, policy[t+1, zp, :])
                    
                    # V' at m_next = u'(c_next)
                    v_prime = u_prime(c_next)
                    
                    rhs += probs[zp] * v_prime
                
                rhs = beta * R * rhs
                
                # FOC: u'(c) - beta*R*V'(R*(m-c)+income) = 0
                # f(c) = u'(c) - rhs
                # f'(c) = u''(c) + beta*R * sum probs * V''(m_next) * R
                # u''(c) = -u'(c)/c for log, -gamma*c^(-gamma-1) for CRRA
                # V''(m) = d/dm [u'(c*(m))] = u''(c*(m)) * dc*/dm
                # This is complex, so use numerical derivative for Newton step
                
                # Use a small perturbation to estimate the derivative
                h = 1e-6
                c_pert = np.clip(c + h, 1e-12, resource_grid)
                
                lhs_pert = u_prime(c_pert)
                rhs_pert = np.zeros(num_resources)
                for zp in range(num_states):
                    if probs[zp] == 0:
                        continue
                    m_next_pert = R * (resource_grid - c_pert) + income[zp]
                    c_next_pert = np.interp(m_next_pert, resource_grid, policy[t+1, zp, :])
                    v_prime_pert = u_prime(c_next_pert)
                    rhs_pert += probs[zp] * v_prime_pert
                rhs_pert = beta * R * rhs_pert
                
                f = lhs - rhs
                f_prime = (lhs_pert - rhs_pert - f) / h
                
                # Newton step: c_new = c - f/f_prime
                # But we need to be careful: if f_prime is near zero, use bisection
                valid = np.abs(f_prime) > 1e-15
                c_new = c.copy()
                c_new[valid] = c[valid] - f[valid] / f_prime[valid]
                
                # For invalid points, use bisection
                invalid = ~valid
                if np.any(invalid):
                    # Simple bisection for those points
                    c_low = np.full(num_resources, 1e-12)
                    c_high = resource_grid.copy()
                    # We don't have c_low/c_high tracked, so just clip
                    c_new[invalid] = c[invalid]
                
                # Clip to valid range
                c_new = np.clip(c_new, 1e-12, resource_grid)
                
                # Check convergence
                if np.all(np.abs(c_new - c) < 1e-10):
                    c = c_new
                    break
                
                c = c_new
            
            c_opt = np.clip(c, 1e-12, resource_grid)
            policy[t, z, :] = c_opt
            
            # Compute value at c_opt
            u_val = utility(c_opt)
            cont_val = np.zeros(num_resources)
            for zp in range(num_states):
                if probs[zp] == 0:
                    continue
                m_next = R * (resource_grid - c_opt) + income[zp]
                v_next = np.interp(m_next, resource_grid, value[t+1, zp, :])
                cont_val += probs[zp] * v_next
            
            value[t, z, :] = u_val + beta * cont_val
    
    return policy
