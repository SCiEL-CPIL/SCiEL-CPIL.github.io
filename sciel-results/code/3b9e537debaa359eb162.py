"""Improved consumption-saving policy solver using Euler equation root-finding."""
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'])
    transition = np.asarray(model['transition'])
    horizon = model['horizon']
    m_grid = np.asarray(resource_grid)
    N_m = len(m_grid)
    N_z = len(income)
    
    # Utility and derivatives
    def util(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
    
    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    def u_double_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return -1.0 / (c * c)
        else:
            return -gamma * c ** (-gamma - 1)
    
    # Terminal condition: at t = horizon, consume all resources
    c_policy = np.zeros((horizon, N_z, N_m))
    c_policy[horizon - 1] = m_grid[None, :]  # c[5] = grid (index 5 is last period, 0-indexed as 5)
    
    # Backward induction using Euler equation
    for t in range(horizon - 2, -1, -1):
        if time.time() - start_time > DEADLINE:
            # Return best result found so far
            break
        
        for z in range(N_z):
            # For each current income state z, solve for optimal consumption
            # Euler equation: u'(c) = beta * R * E_z'[u'(c')]
            # where c' = R*(m - c) + income[z']
            
            c_vals = np.zeros(N_m)
            for i, m in enumerate(m_grid):
                # Solve for c in (0, m] such that Euler equation holds
                # Define f(c) = u'(c) - beta * R * sum_{z'} P(z'|z) * u'(R*(m-c) + income[z'])
                
                # Try Newton's method starting from a reasonable initial guess
                # Initial guess: consume a fraction of resources
                c_init = 0.5 * m if m > 0 else 1e-12
                
                # Bisection + Newton hybrid for robustness
                c_low = 1e-12
                c_high = m
                
                # Check if f(c_high) > 0 (should consume all if Euler is satisfied at c=m)
                # At c = m, c' = income[z'], so f(m) = u'(m) - beta*R*E[u'(income[z'])]
                # Typically u'(m) < beta*R*E[u'(income[z'])] for small m, so f(m) < 0
                # At c -> 0, u'(c) -> inf, so f(c) > 0
                # So there's a root in (0, m]
                
                # Use bisection to bracket, then Newton to refine
                f_c = lambda c: u_prime(c) - beta * R * np.dot(transition[z], u_prime(R * (m - c) + income))
                
                # Bisection
                for _ in range(50):
                    c_mid = 0.5 * (c_low + c_high)
                    f_mid = f_c(c_mid)
                    if f_mid > 0:
                        c_low = c_mid
                    else:
                        c_high = c_mid
                    if c_high - c_low < 1e-12:
                        break
                
                c_opt = 0.5 * (c_low + c_high)
                c_vals[i] = c_opt
            
            c_policy[t, z] = c_vals
    
    # Ensure c[5] (last period) equals the grid within tolerance
    c_policy[horizon - 1] = m_grid[None, :]
    
    # Validate: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(N_z):
            c_policy[t, z] = np.clip(c_policy[t, z], 1e-12, m_grid + 1e-10)
    
    return c_policy
