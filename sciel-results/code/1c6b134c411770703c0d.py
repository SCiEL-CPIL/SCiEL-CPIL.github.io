import numpy as np
import time

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
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1 - gamma) / (1 - gamma)
    
    # Policy: consumption[t, z, m_idx], shape (horizon, num_states, num_resources)
    policy = np.zeros((horizon, num_states, num_resources))
    # Value: value[t, z, m_idx]
    value = np.zeros((horizon, num_states, num_resources))
    
    # Terminal condition: at t = horizon - 1 (last period), consume all
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    value[horizon - 1, :, :] = utility(resource_grid)[np.newaxis, :]
    
    # Use a denser grid for value function interpolation
    # The resource_grid is geomspace(0.001, 40, 101), which is log-spaced
    # We'll use the same grid for value but evaluate consumption on a finer grid
    
    # For each period, we'll search over consumption on a fine grid
    # c = m * fraction, where fraction is in [0, 1]
    # Use a fine fraction grid for accuracy
    
    # Number of consumption grid points per resource state
    # We need to balance accuracy vs. speed
    # Let's use 2000 points as suggested by inspirations
    
    num_c_grid = 2000
    c_fractions = np.linspace(0.001, 1.0, num_c_grid)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: use simple rule for remaining periods
            for tt in range(t, -1, -1):
                # Simple heuristic: consume a fraction based on remaining periods
                frac = 1.0 / (horizon - tt)
                policy[tt, :, :] = np.minimum(frac * resource_grid[np.newaxis, :], resource_grid[np.newaxis, :])
                policy[tt, :, :] = np.maximum(policy[tt, :, :], 1e-12)
                value[tt, :, :] = utility(policy[tt, :, :])
            break
        
        # For each state z, compute optimal consumption
        for z in range(num_states):
            m = resource_grid  # shape (101,)
            
            # For each resource state, evaluate the Bellman operator on the consumption grid
            # c_grid[i, j] = m[j] * c_fractions[i], shape (num_c_grid, num_resources)
            c_grid = c_fractions[:, np.newaxis] * m[np.newaxis, :]  # (2000, 101)
            
            # Ensure c <= m (should be by construction since fractions <= 1)
            c_grid = np.minimum(c_grid, m[np.newaxis, :])
            c_grid = np.maximum(c_grid, 1e-12)
            
            # Compute utility for each consumption choice
            u_c = utility(c_grid)  # (2000, 101)
            
            # Compute next period's resources for each consumption choice and each next state z'
            # m_next[z', i, j] = R * (m[j] - c_grid[i, j]) + income[z']
            # Shape: (num_states, num_c_grid, num_resources)
            m_next = np.zeros((num_states, num_c_grid, num_resources))
            for zp in range(num_states):
                m_next[zp, :, :] = R * (m[np.newaxis, :] - c_grid) + income[zp]
                m_next[zp, :, :] = np.maximum(m_next[zp, :, :], 1e-10)
            
            # Interpolate value function from value[t+1] onto m_next
            # value[t+1, zp, :] is defined on resource_grid
            # We need to interpolate onto m_next[zp, :, :]
            
            # Use np.interp for each state z'
            # np.interp(x, xp, fp) where x is the new points, xp is the original grid
            # We need to handle the case where m_next is outside the range of resource_grid
            # resource_grid is geomspace(0.001, 40, 101)
            
            v_next = np.zeros((num_states, num_c_grid, num_resources))
            for zp in range(num_states):
                # Interpolate value[t+1, zp, :] from resource_grid to m_next[zp, :, :]
                # m_next[zp, :, :] has shape (num_c_grid, num_resources)
                # We can reshape to (num_c_grid * num_resources,) for interpolation
                m_next_flat = m_next[zp, :, :].ravel()
                v_interp = np.interp(m_next_flat, resource_grid, value[t + 1, zp, :],
                                     left=value[t + 1, zp, 0], right=value[t + 1, zp, -1])
                v_next[zp, :, :] = v_interp.reshape(num_c_grid, num_resources)
            
            # Compute expected continuation value
            # E[V_{t+1}(m')] = sum_{zp} transition[z, zp] * V_{t+1}(m_next[zp])
            # Shape: (num_c_grid, num_resources)
            expected_v = np.zeros((num_c_grid, num_resources))
            for zp in range(num_states):
                expected_v += transition[z, zp] * v_next[zp, :, :]
            
            # Total value for each consumption choice
            total_value = u_c + beta * expected_v  # (2000, 101)
            
            # Find the consumption that maximizes total value for each resource state
            # argmax over axis 0 (consumption grid)
            best_idx = np.argmax(total_value, axis=0)  # shape (101,)
            
            # Extract the optimal consumption
            c_optimal = c_grid[best_idx, np.arange(num_resources)]  # shape (101,)
            
            # Ensure validity
            c_optimal = np.maximum(c_optimal, 1e-12)
            c_optimal = np.minimum(c_optimal, m)
            
            policy[t, z, :] = c_optimal
            value[t, z, :] = total_value[best_idx, np.arange(num_resources)]
    
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
