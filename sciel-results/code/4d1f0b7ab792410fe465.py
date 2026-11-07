import numpy as np
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    trans = model['transition']
    horizon = model['horizon']
    
    # Grid
    m_grid = resource_grid
    n_m = len(m_grid)
    n_z = len(income)
    
    # Policy storage: c[t, z, m]
    c_policy = np.zeros((horizon, n_z, n_m))
    
    # Terminal period
    c_policy[horizon-1, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    def util(c, gamma):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        return np.power(c, 1.0 - gamma) / (1.0 - gamma)
        
    def mu(c, gamma):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        return np.power(c, -gamma)
        
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        c_next = c_policy[t+1] # Shape (n_z, n_m)
        
        for z in range(n_z):
            for i_m, m in enumerate(m_grid):
                # Function to evaluate marginal utility of next period's consumption
                # given current consumption c
                def euler_residual(c):
                    if c <= 0: return mu(1e-12, gamma) # Large positive
                    if c >= m: 
                         # Check if saving is 0
                         # m' = R(0) + income[z']
                         # We need to see if g(m) > 0 or < 0
                         pass
                     
                    m_savings = R * (m - c)
                    target_mu = 0.0
                    for z_next in range(n_z):
                        p = trans[z, z_next]
                        if p < 1e-10: continue
                        m_next = m_savings + income[z_next]
                        # Interpolate c_next for state z_next at m_next
                        # Clip m_next to grid range for interpolation
                        m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])
                        c_next_val = np.interp(m_next_clipped, m_grid, c_next[z_next, :])
                        target_mu += p * mu(c_next_val, gamma)
                    
                    target_mu *= beta * R
                    return mu(c, gamma) - target_mu
                
                # Solve for c in [1e-12, m]
                # Check endpoints
                res_low = euler_residual(1e-12)
                res_high = euler_residual(m)
                
                c_opt = m # Default
                if res_low * res_high < 0:
                    c_opt = brentq(euler_residual, 1e-12, m, xtol=1e-12, rtol=1e-12)
                elif res_low < 0:
                    c_opt = 1e-12
                else:
                    c_opt = m
                    
                c_policy[t, z, i_m] = c_opt
                
    return c_policy
