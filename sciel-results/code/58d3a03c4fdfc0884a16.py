import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'])
    transition = np.asarray(model['transition'])
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    
    # Policy: consumption[t, z, m_idx], shape (horizon, num_states, num_resources)
    policy = np.zeros((horizon, num_states, num_resources))
    # Value: value[t, z, m_idx]
    value = np.zeros((horizon, num_states, num_resources))
    
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1.0 - gamma)) / (1.0 - gamma)
    
    # Final period: consume everything
    for z in range(num_states):
        policy[horizon - 1, z, :] = resource_grid
        value[horizon - 1, z, :] = utility(resource_grid)
    
    # Backward induction for t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break
        for z in range(num_states):
            # For each resource level, find optimal consumption
            # m_next = R * (m - c) + income[z']
            # We need to maximize: utility(c) + beta * sum_{z'} transition[z, z'] * V[t+1, z', m_next]
            
            # Use a grid of consumption fractions
            # c = m * frac, where frac is in [0, 1]
            # m_next = R * m * (1 - frac) + income[z']
            
            # For efficiency, we'll use interpolation on the next period's value
            # and evaluate on a grid of consumption levels
            
            # Create a fine grid for consumption as fraction of m
            # But m varies, so we need to handle this carefully
            
            # Strategy: for each m_idx, compute optimal c
            # Use np.interp for continuation values
            
            m_vals = resource_grid
            optimal_c = np.zeros(num_resources)
            
            # We'll use a moderate number of candidate consumption levels per m
            # c ranges from 0 to m
            # Let's use fractions: c = m * frac, frac in linspace(0, 1, N)
            n_frac = min(budget, 200)
            fracs = np.linspace(0.001, 1.0, n_frac)
            
            for m_idx in range(num_resources):
                m = m_vals[m_idx]
                if m < 1e-10:
                    optimal_c[m_idx] = 1e-12
                    continue
                
                c_candidates = m * fracs
                c_candidates = np.maximum(c_candidates, 1e-12)
                
                best_val = -np.inf
                best_c = c_candidates[0]
                
                for z_prime in range(num_states):
                    if transition[z, z_prime] < 1e-15:
                        continue
                    
                    m_next = R * (m - c_candidates) + income[z_prime]
                    m_next = np.maximum(m_next, 0)
                    
                    # Interpolate value[t+1, z_prime] at m_next
                    # value[t+1, z_prime] is defined on resource_grid
                    # We need to handle m_next outside the grid
                    v_cont = np.interp(m_next, resource_grid, value[t + 1, z_prime], 
                                       left=value[t + 1, z_prime, 0], 
                                       right=value[t + 1, z_prime, -1])
                    
                    obj = utility(c_candidates) + beta * transition[z, z_prime] * v_cont
                
                # Find best
                idx_best = np.argmax(obj)
                best_c = c_candidates[idx_best]
                optimal_c[m_idx] = best_c
            
            policy[t, z, :] = optimal_c
            # Compute value at the optimal consumption
            for z_prime in range(num_states):
                if transition[z, z_prime] < 1e-15:
                    continue
                m_next = R * (resource_grid - optimal_c) + income[z_prime]
                m_next = np.maximum(m_next, 0)
                v_cont = np.interp(m_next, resource_grid, value[t + 1, z_prime],
                                   left=value[t + 1, z_prime, 0],
                                   right=value[t + 1, z_prime, -1])
                value[t, z, :] += beta * transition[z, z_prime] * v_cont
            value[t, z, :] += utility(np.maximum(optimal_c, 1e-12))
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_states):
            policy[t, z, :] = np.maximum(policy[t, z, :], 1e-12)
            policy[t, z, :] = np.minimum(policy[t, z, :], resource_grid + 1e-10)
    
    # Final period must equal grid
    for z in range(num_states):
        policy[horizon - 1, z, :] = resource_grid
    
    return policy
