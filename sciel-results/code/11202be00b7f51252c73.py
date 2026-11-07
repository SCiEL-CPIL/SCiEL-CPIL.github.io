import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110
    
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
    
    # Precompute utility and marginal utility functions
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-9:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
    
    def marginal_utility(c):
        c = np.maximum(c, 1e-12)
        return c ** (-gamma)
    
    # Work backward from t = horizon-2 to t = 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fill remaining with terminal-like policy
            for tt in range(t, horizon - 1):
                policy[tt, :, :] = resource_grid
            break
        
        # For each income state z, solve the policy for all m values
        for z in range(2):
            # Precompute the expected continuation marginal utility as a function of c
            # For each m, we need to find c that satisfies the Euler equation
            # u'(c) = beta * R * E[u'(c_next)]
            # where c_next = policy[t+1, z', R*(m-c) + income[z']]
            
            # We'll use a vectorized approach with Newton's method
            m_values = resource_grid
            
            # Initialize with a reasonable guess: c = m * 0.5
            c = m_values * 0.5
            c = np.maximum(c, 1e-12)
            c = np.minimum(c, m_values)
            
            # Newton's method iterations
            for iteration in range(50):
                if time.time() > deadline:
                    break
                    
                # Compute m_next for each z'
                m_next = R * (m_values - c)  # shape (n_out,)
                
                # Compute c_next for each z'
                c_next_0 = np.interp(m_next + income[0], resource_grid, policy[t + 1, 0, :])
                c_next_1 = np.interp(m_next + income[1], resource_grid, policy[t + 1, 1, :])
                
                # Clip c_next to valid range
                c_next_0 = np.clip(c_next_0, 1e-12, resource_grid)
                c_next_1 = np.clip(c_next_1, 1e-12, resource_grid)
                
                # Compute marginal utilities
                mu_c = marginal_utility(c)  # shape (n_out,)
                mu_next_0 = marginal_utility(c_next_0)
                mu_next_1 = marginal_utility(c_next_1)
                
                # Expected marginal utility
                E_mu_next = transition[z, 0] * mu_next_0 + transition[z, 1] * mu_next_1
                
                # Residual: u'(c) - beta * R * E[u'(c_next)]
                residual = mu_c - beta * R * E_mu_next
                
                # Check convergence
                if np.max(np.abs(residual)) < 1e-12:
                    break
                
                # Compute derivative of residual with respect to c
                # d/dc [u'(c)] = -gamma * c^(-gamma-1)
                d_mu_c_dc = -gamma * c ** (-gamma - 1)
                
                # d/dc [E_mu_next] = E[d_mu_next/dc_next * dc_next/dc]
                # dc_next/dc = -R (since m_next = R*(m-c), so dm_next/dc = -R)
                # d_mu_next/dc_next = -gamma * c_next^(-gamma-1)
                d_mu_next_0_dc = -gamma * c_next_0 ** (-gamma - 1) * (-R)
                d_mu_next_1_dc = -gamma * c_next_1 ** (-gamma - 1) * (-R)
                
                d_E_mu_next_dc = transition[z, 0] * d_mu_next_0_dc + transition[z, 1] * d_mu_next_1_dc
                
                d_residual_dc = d_mu_c_dc - beta * R * d_E_mu_next_dc
                
                # Newton step
                # Avoid division by zero
                d_residual_dc = np.where(np.abs(d_residual_dc) < 1e-12, 1e-12, d_residual_dc)
                delta_c = residual / d_residual_dc
                
                # Damped Newton step
                c_new = c - delta_c
                c_new = np.clip(c_new, 1e-12, m_values)
                
                # Check if we're making progress
                if np.max(np.abs(c_new - c)) < 1e-14:
                    c = c_new
                    break
                c = c_new
            
            policy[t, z, :] = c
    
    # Ensure validity
    for t in range(horizon):
        for z in range(2):
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, resource_grid + 1e-10)
    
    # Terminal period must equal grid
    policy[horizon - 1, :, :] = resource_grid
    
    return policy
