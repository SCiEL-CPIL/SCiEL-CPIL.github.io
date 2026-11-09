"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar
import time

def solve_consumption(model, resource_grid, budget):
    t_start = time.time()
    deadline = 145.0  # safety margin before 150s
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)
    n_grid = len(resource_grid)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
    
    def utility_scalar(c):
        c = max(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
    
    # Policy: consumption[t, z, grid_index]
    policy = np.zeros((horizon, n_states, n_grid))
    
    # Last period: consume all
    policy[horizon - 1, :, :] = resource_grid[None, :]
    
    # Backward induction from t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        # Check time
        if time.time() > deadline - 10.0:
            # Fallback: simple rule
            for z in range(n_states):
                policy[t, z, :] = np.minimum(resource_grid, resource_grid * 0.5)
            break
        
        for z in range(n_states):
            for i in range(n_grid):
                m = resource_grid[i]
                
                # Define the objective function to maximize
                def objective(log_c):
                    c = np.exp(log_c)
                    if c > m:
                        c = m
                    if c < 1e-12:
                        c = 1e-12
                    
                    # Current utility
                    obj = utility_scalar(c)
                    
                    # Expected future utility
                    future_val = 0.0
                    for zp in range(n_states):
                        prob = transition[z, zp]
                        if prob < 1e-15:
                            continue
                        m_next = R * (m - c) + income[zp]
                        m_next = max(m_next, 1e-12)
                        # Interpolate next period's policy
                        c_next = np.interp(m_next, resource_grid, policy[t + 1, zp, :])
                        c_next = max(c_next, 1e-12)
                        future_val += prob * utility_scalar(c_next)
                    
                    obj += beta * future_val
                    return -obj  # Negate for minimization
                
                # Solve for c in log space: c = exp(x), x in [log(1e-12), log(m)]
                x_lo = np.log(1e-12)
                x_hi = np.log(max(m, 1e-12))
                
                try:
                    result = minimize_scalar(objective, bounds=(x_lo, x_hi), method='bounded',
                                             options={'xatol': 1e-10, 'maxiter': 100})
                    c_sol = np.exp(result.x)
                    c_sol = max(1e-12, min(c_sol, m))
                except Exception:
                    c_sol = m * 0.5
                
                policy[t, z, i] = c_sol
    
    # Ensure constraints
    for t in range(horizon):
        for z in range(n_states):
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, resource_grid + 1e-10)
    
    # Final period must equal grid
    policy[horizon - 1, :, :] = resource_grid[None, :]
    
    return policy
# EVOLVE-BLOCK-END
