import numpy as np
import time
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar

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
            for i, m in enumerate(resource_grid):
                if time.time() > deadline:
                    policy[t, z, i:] = resource_grid[i:]
                    break
                
                # Define objective to maximize: utility(c) + beta * E[continuation_value]
                def objective(c):
                    m_next = R * (m - c) + income  # shape (2,)
                    c_next = np.array([
                        float(pchip_policies[t + 1][0](m_next[0])),
                        float(pchip_policies[t + 1][1](m_next[1]))
                    ])
                    u_c = utility(c)
                    u_next = utility(c_next)
                    return -(u_c + beta * np.dot(transition[z], u_next))
                
                # Bounds: c in [1e-12, m]
                lo = 1e-12
                hi = m
                
                try:
                    result = minimize_scalar(objective, bounds=(lo, hi), method='bounded', 
                                           options={'maxiter': 100, 'xatol': 1e-12})
                    if result.success:
                        policy[t, z, i] = result.x
                    else:
                        # Fallback: grid search
                        fractions = np.linspace(0.001, 1.0, 100)
                        c_candidates = m * fractions
                        best_c = c_candidates[0]
                        best_val = np.inf
                        for c in c_candidates:
                            val = objective(c)
                            if val < best_val:
                                best_val = val
                                best_c = c
                        policy[t, z, i] = best_c
                except Exception:
                    # Fallback: grid search
                    fractions = np.linspace(0.001, 1.0, 100)
                    c_candidates = m * fractions
                    best_c = c_candidates[0]
                    best_val = np.inf
                    for c in c_candidates:
                        val = objective(c)
                        if val < best_val:
                            best_val = val
                            best_c = c
                    policy[t, z, i] = best_c
        
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
