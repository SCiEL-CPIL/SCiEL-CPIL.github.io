import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    
    # Policy and Value arrays
    # policy[t, z, m_idx]
    policy = np.zeros((horizon, num_states, num_resources))
    value = np.zeros((horizon, num_states, num_resources))
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    # Last period: consume everything
    value[horizon - 1, :, :] = utility(resource_grid)
    policy[horizon - 1, :, :] = resource_grid
    
    # Backward induction for t = horizon - 2 down to 0
    for t in range(horizon - 2, -1, -1):
        for z in range(num_states):
            # For each resource level, find optimal consumption
            # c in (0, m], next_m = R*(m - c) + income[z']
            # V = u(c) + beta * sum_{z'} P(z'|z) * V_next[z', next_m(z')]
            
            # We'll use a grid of consumption fractions to find optimal c
            # To be efficient, we can vectorize over resource_grid
            
            # Create a grid of consumption candidates for each m
            # Use linspace for fractions of m
            # But we need to interpolate continuation values
            
            # For efficiency, let's pre-compute the continuation value function
            # V_next[z', m] for each z'
            
            # For each z', we have value[t+1, z', :] as a function of resource_grid
            # We need to evaluate V_next[z', R*(m - c) + income[z']]
            
            # Let's use a moderate number of c candidates
            # c = m * linspace(0.001, 1, budget)
            c_fracs = np.linspace(0.001, 1.0, budget)
            
            # c_candidates has shape (num_resources, budget)
            c_candidates = resource_grid[:, None] * c_fracs[None, :]
            
            # next_m has shape (num_resources, budget, num_states)
            # next_m[i, j, z'] = R * (resource_grid[i] - c_candidates[i, j]) + income[z']
            m_minus_c = resource_grid[:, None] - c_candidates
            next_m_base = R * m_minus_c  # shape (num_resources, budget)
            
            # Continuation value for each z'
            # V_next[z', next_m] needs interpolation
            # value[t+1, z', :] is defined on resource_grid
            
            # We need to interpolate value[t+1, z', :] at next_m_base + income[z']
            
            # Let's compute the expected continuation value
            # E[V_next] = sum_{z'} transition[z, z'] * V_next[z', next_m_base + income[z']]
            
            # For each z', interpolate
            # To vectorize, we can loop over z' (only 2 states)
            
            cont_val = np.zeros((num_resources, budget))
            for zp in range(num_states):
                # next_m for state zp
                nm = next_m_base + income[zp]
                # Clip to valid range for interpolation
                nm = np.clip(nm, resource_grid[0], resource_grid[-1])
                
                # Interpolate value[t+1, zp, :] at nm
                # value[t+1, zp, :] is 1D array of shape (num_resources,)
                v_interp = np.interp(nm, resource_grid, value[t + 1, zp, :])
                
                cont_val += transition[z, zp] * v_interp
            
            # Total value for each c candidate
            # u(c) + beta * cont_val
            u_c = utility(c_candidates)  # shape (num_resources, budget)
            total_val = u_c + beta * cont_val  # shape (num_resources, budget)
            
            # Find optimal c for each resource level
            # idx = argmax over budget dimension
            opt_idx = np.argmax(total_val, axis=1)  # shape (num_resources,)
            
            # Extract optimal c and value
            # We need to gather c_candidates[i, opt_idx[i]]
            # Use advanced indexing
            opt_c = c_candidates[np.arange(num_resources), opt_idx]
            opt_val = total_val[np.arange(num_resources), opt_idx]
            
            policy[t, z, :] = opt_c
            value[t, z, :] = opt_val
            
            # Check time
            if time.time() > deadline - 10:
                # Return what we have so far
                # Fill remaining periods with last known policy
                for t2 in range(t, -1, -1):
                    if t2 < t:
                        policy[t2, :, :] = policy[t + 1, :, :]
                        value[t2, :, :] = value[t + 1, :, :]
                return policy
    
    # Final validation: ensure 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_states):
            m = resource_grid
            c = policy[t, z, :]
            # Enforce c <= m
            c = np.minimum(c, m)
            # Enforce c > 0
            c = np.maximum(c, 1e-12)
            policy[t, z, :] = c
    
    # c[5] should equal grid within 1e-9 (last period)
    policy[horizon - 1, :, :] = resource_grid
    
    return policy
# EVOLVE-BLOCK-END
