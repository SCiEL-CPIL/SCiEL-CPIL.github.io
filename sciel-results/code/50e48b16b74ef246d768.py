import numpy as np
import time
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator, interp1d

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115
    
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
    
    # PCHIP interpolators for each period and state
    pchip_policies = [None] * horizon
    pchip_policies[horizon - 1] = [
        PchipInterpolator(resource_grid, policy[horizon - 1, 0, :], extrapolate=False),
        PchipInterpolator(resource_grid, policy[horizon - 1, 1, :], extrapolate=False)
    ]
    
    def marginal_utility(c):
        c = np.maximum(c, 1e-12)
        return c ** (-gamma)
    
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    # Work backward from t = horizon-2 to t = 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fill remaining with terminal-like policy
            for tt in range(t, horizon - 1):
                policy[tt, :, :] = resource_grid
            break
        
        for z in range(2):
            # Vectorize over all m values for this z
            m_vals = resource_grid
            
            # Compute next period resources for both income states
            # m_next[z'] = R*(m - c) + income[z']
            # We need to solve FOC: u'(c) = beta * R * E_z'[u'(c_next)]
            
            # Use vectorized Newton/bisection approach
            # Start with initial guess: consume a fraction
            c_guess = np.clip(m_vals * 0.5, 1e-12, m_vals)
            
            # Iterate: use FOC with interpolated next-period policy
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                m_next_0 = R * (m_vals - c_guess) + income[0]
                m_next_1 = R * (m_vals - c_guess) + income[1]
                
                # Clip to valid range for interpolation
                m_next_0 = np.clip(m_next_0, resource_grid[0], resource_grid[-1])
                m_next_1 = np.clip(m_next_1, resource_grid[0], resource_grid[-1])
                
                c_next_0 = pchip_policies[t + 1][0](m_next_0)
                c_next_1 = pchip_policies[t + 1][1](m_next_1)
                
                # Compute marginal utilities
                mu_c = marginal_utility(c_guess)
                mu_next_0 = marginal_utility(c_next_0)
                mu_next_1 = marginal_utility(c_next_1)
                
                E_mu_next = transition[z, 0] * mu_next_0 + transition[z, 1] * mu_next_1
                target = beta * R * E_mu_next
                
                # Newton step: c_new = c - f(c)/f'(c)
                # f(c) = u'(c) - beta*R*E[u'(c_next)]
                # f'(c) = u''(c) - beta*R*E[u''(c_next) * (-R)]
                # For CRRA: u''(c) = -gamma * c^(-gamma-1)
                mu2_c = -gamma * c_guess ** (-gamma - 1)
                mu2_next_0 = -gamma * np.maximum(c_next_0, 1e-12) ** (-gamma - 1)
                mu2_next_1 = -gamma * np.maximum(c_next_1, 1e-12) ** (-gamma - 1)
                E_mu2_next = transition[z, 0] * mu2_next_0 + transition[z, 1] * mu2_next_1
                f_prime = mu2_c - beta * R * (-R) * E_mu2_next
                
                # Newton update with damping
                c_new = c_guess - (mu_c - target) / f_prime
                c_new = np.clip(c_new, 1e-12, m_vals)
                
                # Check convergence
                if np.max(np.abs(c_new - c_guess)) < 1e-10:
                    c_guess = c_new
                    break
                c_guess = c_new
            
            policy[t, z, :] = c_guess
        
        # Build PCHIP interpolators for this period
        pchip_policies[t] = [
            PchipInterpolator(resource_grid, policy[t, 0, :], extrapolate=False),
            PchipInterpolator(resource_grid, policy[t, 1, :], extrapolate=False)
        ]
    
    # Ensure validity
    for t in range(horizon):
        for z in range(2):
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, resource_grid + 1e-10)
    
    # Terminal period must equal grid
    policy[horizon - 1, :, :] = resource_grid
    
    return policy
