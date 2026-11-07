import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_out = len(resource_grid)
    
    # Policy shape: (horizon, 2, n_out)
    policy = np.zeros((horizon, 2, n_out))
    
    # Terminal period: consume everything
    policy[horizon - 1, :, :] = resource_grid
    
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-9:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
    
    def marginal_utility(c):
        c = np.maximum(c, 1e-12)
        return c ** (-gamma)
    
    # For each period and state, we'll store the consumption policy
    # and use vectorized operations where possible
    
    # Work backward from t = horizon-2 to t = 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fill remaining with terminal-like policy
            for tt in range(t, horizon - 1):
                policy[tt, :, :] = resource_grid
            break
        
        for z in range(2):
            # We need to solve for c[m] for each m in resource_grid
            # Using the Euler equation: u'(c) = beta * R * E[u'(c')]
            # where c' is the optimal consumption next period
            
            # For a given m, the next period's resources are:
            # m_next[z'] = R * (m - c) + income[z']
            # c_next[z'] = policy[t+1, z', m_next[z']]
            
            # The FOC is:
            # u'(c) = beta * sum_z' transition[z, z'] * u'(policy[t+1, z', R*(m-c) + income[z']])
            
            # We can use a fixed-point iteration or Newton's method on the grid
            # Since the policy is monotonic, we can use a simple approach:
            # Start with c = m (consume all), then iterate
            
            # Initialize with a reasonable guess
            c_current = np.maximum(resource_grid * 0.5, 1e-8)
            
            # Iterate to convergence
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                # Compute next period resources for each grid point
                # m_next has shape (2, n_out) for the two possible states
                m_next = R * (resource_grid[None, :] - c_current[None, :]) + income[:, None]
                
                # Get next period consumption for each state
                # We need to interpolate policy[t+1] at m_next
                c_next = np.zeros((2, n_out))
                for z_prime in range(2):
                    # Clip m_next to valid range
                    m_next_clipped = np.clip(m_next[z_prime, :], resource_grid[0], resource_grid[-1])
                    c_next[z_prime, :] = np.interp(m_next_clipped, resource_grid, policy[t + 1, z_prime, :])
                
                # Compute expected marginal utility of next period consumption
                # E[u'(c')] = sum_z' transition[z, z'] * u'(c_next[z'])
                mu_next = marginal_utility(c_next)  # shape (2, n_out)
                expected_mu_next = transition[z, 0] * mu_next[0, :] + transition[z, 1] * mu_next[1, :]
                
                # FOC: u'(c) = beta * expected_mu_next
                # u'(c) = c^(-gamma)
                # c^(-gamma) = beta * expected_mu_next
                # c = (beta * expected_mu_next)^(-1/gamma)
                
                c_new = (beta * expected_mu_next) ** (-1.0 / gamma)
                
                # Ensure c is within bounds [1e-12, m]
                c_new = np.clip(c_new, 1e-12, resource_grid)
                
                # Check convergence
                if np.max(np.abs(c_new - c_current)) < 1e-10:
                    c_current = c_new
                    break
                
                c_current = c_new
            
            policy[t, z, :] = c_current
    
    # Ensure validity
    for t in range(horizon):
        for z in range(2):
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, resource_grid + 1e-10)
    
    # Terminal period must equal grid
    policy[horizon - 1, :, :] = resource_grid
    
    return policy
