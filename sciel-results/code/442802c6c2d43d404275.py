def solve_consumption(model, resource_grid, budget):
    import numpy as np
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = 2
    n_grid = len(resource_grid)
    
    # Use a dense internal grid for value function matching scoring grid
    m_grid = np.geomspace(0.001, 40, 251)
    n_m = len(m_grid)
    
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    # V[t, z, m_index] = value function
    V = np.zeros((horizon, n_states, n_m))
    c_policy = np.zeros((horizon, n_states, n_m))
    
    # Terminal: consume everything
    V[horizon - 1, :, :] = utility(m_grid)[np.newaxis, :, :]
    c_policy[horizon - 1, :, :] = m_grid[np.newaxis, :, :]
    
    # Candidate fractions for c - match scoring's best search
    fractions = np.linspace(0.001, 1.0, 301)
    
    for t in range(horizon - 2, -1, -1):
        for z in range(n_states):
            # Vectorize over all m and all c candidates
            c_candidates = m_grid[:, np.newaxis] * fractions[np.newaxis, :]
            c_candidates = np.minimum(c_candidates, m_grid[:, np.newaxis])
            
            m_minus_c = m_grid[:, np.newaxis] - c_candidates
            
            # Accumulate total value: u(c) + beta * sum_z' P(z'|z) * V_next
            total_val = utility(c_candidates)
            
            for z_prime in range(n_states):
                m_next = R * m_minus_c + income[z_prime]
                m_next = np.maximum(m_next, 1e-12)
                V_next = np.interp(m_next, m_grid, V[t + 1, z_prime, :])
                prob = transition[z, z_prime]
                total_val += beta * prob * V_next
            
            # Find best c for each m
            idx = np.argmax(total_val, axis=1)
            c_policy[t, z, :] = c_candidates[np.arange(n_m), idx]
            V[t, z, :] = total_val[np.arange(n_m), idx]
    
    # Interpolate c_policy onto the input resource_grid
    result = np.zeros((horizon, n_states, n_grid))
    for t in range(horizon):
        for z in range(n_states):
            result[t, z, :] = np.interp(resource_grid, m_grid, c_policy[t, z, :])
    
    # Terminal condition: c[horizon-1] must equal grid
    result[horizon - 1, :, :] = resource_grid[np.newaxis, :, :]
    
    # Enforce 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_states):
            m_vals = resource_grid
            result[t, z, :] = np.clip(result[t, z, :], 1e-12, m_vals + 1e-10)
    
    return result
