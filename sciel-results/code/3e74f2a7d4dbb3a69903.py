import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
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
    
    # Policy and Value arrays
    # policy[t, z, m_idx]
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
                
                # Find optimal consumption using golden section search
                # c is in [0, m]
                # We want to maximize: utility(c) + beta * sum_z' transition[z, z'] * V[t+1, z', R*(m-c) + income[z']]
                
                # Define the objective function
                def objective(c):
                    if c < 0 or c > m + 1e-10:
                        return -np.inf
                    c = max(c, 1e-12)
                    c = min(c, m)
                    
                    # Continuation value
                    cont_val = 0.0
                    for zp in range(num_states):
                        m_next = R * (m - c) + income[zp]
                        # Interpolate value at m_next
                        if m_next <= resource_grid[0]:
                            v_next = value[t + 1, zp, 0]
                        elif m_next >= resource_grid[-1]:
                            v_next = value[t + 1, zp, -1]
                        else:
                            v_next = np.interp(m_next, resource_grid, value[t + 1, zp, :])
                        cont_val += transition[z, zp] * v_next
                    
                    return utility(c) + beta * cont_val
                
                # Golden section search for maximum
                a = 0.0
                b = m
                
                # Golden ratio
                gr = (np.sqrt(5.0) - 1.0) / 2.0
                c1 = b - gr * (b - a)
                c2 = a + gr * (b - a)
                
                f1 = objective(c1)
                f2 = objective(c2)
                
                # Check deadline
                if time.time() > deadline:
                    # Return best found so far
                    break
                
                # Iterate golden section search
                for _ in range(50):
                    if f1 > f2:
                        b = c2
                        c2 = c1
                        f2 = f1
                        c1 = b - gr * (b - a)
                        f1 = objective(c1)
                    else:
                        a = c1
                        c1 = c2
                        f1 = f2
                        c2 = a + gr * (b - a)
                        f2 = objective(c2)
                    
                    if abs(b - a) < 1e-10:
                        break
                    
                    if time.time() > deadline:
                        break
                
                # Optimal c is at the midpoint of the final interval
                c_opt = (a + b) / 2.0
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
