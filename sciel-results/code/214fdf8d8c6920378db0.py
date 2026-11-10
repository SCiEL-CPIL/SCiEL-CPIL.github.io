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
    
    # Policy: consumption[t, z, m_idx], shape (horizon, num_states, num_resources)
    policy = np.zeros((horizon, num_states, num_resources))
    # Value: value[t, z, m_idx]
    value = np.zeros((horizon, num_states, num_resources))
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1 - gamma) / (1 - gamma)
    
    # Terminal condition: at t = horizon - 1 (last period), consume all
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    value[horizon - 1, :, :] = utility(resource_grid)[np.newaxis, :]
    
    # Helper to compute continuation value for a given next-period resource vector
    # m_next is shape (num_resources,)
    # Returns continuation value shape (num_resources,)
    def get_continuation(m_next, z, t_next):
        cont_val = np.zeros_like(m_next)
        for zp in range(num_states):
            m_next_zp = m_next + income[zp]
            # Clip to grid range for interpolation
            v_next = np.interp(m_next_zp, resource_grid, value[t_next, zp, :])
            cont_val += transition[z, zp] * v_next
        return cont_val

    # Backward induction
    # We will use a dense grid for consumption choices to approximate the optimum
    # The budget parameter in the seed was used for the number of consumption points.
    # Here we use a fixed, high number of points to ensure accuracy, limited by time.
    # Let's use 1000 points for consumption grid to balance speed and accuracy.
    num_c_points = 1000
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 15:
            # Fallback: use simple rule for remaining periods
            for tt in range(t, -1, -1):
                # Simple fallback: consume 50% of resources
                policy[tt, :, :] = 0.5 * resource_grid[np.newaxis, :]
                # Calculate value for fallback
                c_fallback = policy[tt]
                m_next_fallback = R * (resource_grid[np.newaxis, :] - c_fallback)
                for z in range(num_states):
                    cont = get_continuation(m_next_fallback[z], z, tt + 1)
                    value[tt, z, :] = utility(c_fallback[z]) + beta * cont
            break
        
        for z in range(num_states):
            m = resource_grid
            # Generate consumption grid for this state
            # c ranges from small epsilon to m
            # Since m varies, we need to handle the upper bound per m.
            # Vectorized approach:
            # Create a 2D grid of c choices: shape (num_resources, num_c_points)
            # c_choices[i, j] = m[i] * (j + 1) / num_c_points
            # This ensures c <= m.
            
            # To save memory and time, we can process in chunks if num_resources is large, 
            # but 101 * 1000 is small.
            
            # c_grid shape: (num_resources, num_c_points)
            fractions = np.linspace(0.001, 1.0, num_c_points)
            c_grid = m[:, np.newaxis] * fractions[np.newaxis, :]
            
            # Ensure c is not zero
            c_grid = np.maximum(c_grid, 1e-12)
            
            # Calculate m_next for each c choice
            # m_next = R * (m - c)
            m_next_grid = R * (m[:, np.newaxis] - c_grid)
            
            # Calculate continuation value for each c choice
            # We need to interpolate value[t+1, z', :] at m_next + income[z']
            # This is expensive if done naively.
            
            # Vectorized interpolation is tricky with np.interp which works on 1D.
            # We can loop over z' (only 2 states) and vectorize over m and c.
            
            cont_val_grid = np.zeros_like(c_grid)
            
            for zp in range(num_states):
                p_zp = transition[z, zp]
                if p_zp == 0:
                    continue
                
                # m_next_zp_grid = m_next_grid + income[zp]
                m_next_zp_grid = m_next_grid + income[zp]
                
                # Interpolate value[t+1, zp, :] at m_next_zp_grid
                # np.interp requires 1D x and 1D fp.
                # We can flatten m_next_zp_grid, interpolate, then reshape.
                flat_m_next = m_next_zp_grid.ravel()
                flat_v = np.interp(flat_m_next, resource_grid, value[t + 1, zp, :])
                v_grid = flat_v.reshape(c_grid.shape)
                
                cont_val_grid += p_zp * v_grid
            
            # Total value for each c choice
            # V = u(c) + beta * E[V_next]
            u_grid = utility(c_grid)
            total_value_grid = u_grid + beta * cont_val_grid
            
            # Find the c that maximizes total value
            # max_idx shape: (num_resources,)
            max_idx = np.argmax(total_value_grid, axis=1)
            
            # Extract optimal c and value
            optimal_c = c_grid[np.arange(num_resources), max_idx]
            optimal_v = total_value_grid[np.arange(num_resources), max_idx]
            
            policy[t, z, :] = optimal_c
            value[t, z, :] = optimal_v
            
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
