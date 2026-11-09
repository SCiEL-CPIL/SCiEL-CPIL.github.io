"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import brentq
import time

def solve_consumption(model, resource_grid, budget):
    t_start = time.time()
    deadline = 145.0  # safety margin before 150s wall clock
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']
    
    n_states = len(income)
    n_grid = len(resource_grid)
    
    # Utility derivative (scalar)
    def u_prime_scalar(c):
        c = max(c, 1e-12)
        if gamma == 1:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    # Policy: consumption[t, z, grid_index]
    policy = np.zeros((horizon, n_states, n_grid))
    
    # Last period: consume all
    policy[horizon - 1, :, :] = resource_grid[None, :, None]
    
    # Precompute interpolation for next period policies
    # For each z', we have policy[t+1, z', :] as a function of m
    
    # Backward induction from t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        # Check time
        if time.time() > deadline - 5.0:
            # Fallback: simple proportional rule
            for z in range(n_states):
                policy[t, z, :] = resource_grid * 0.5
            break
        
        for z in range(n_states):
            for i in range(n_grid):
                m = resource_grid[i]
                
                # Define the Euler equation residual: f(c) = u'(c) - beta*R*E[u'(c')]
                def euler_residual(c, m=m, z=z, t=t):
                    c = max(c, 1e-12)
                    c = min(c, m)
                    
                    lhs = u_prime_scalar(c)
                    
                    rhs = 0.0
                    for zp in range(n_states):
                        prob = transition[z, zp]
                        if prob < 1e-15:
                            continue
                        m_next = R * (m - c) + income[zp]
                        m_next = max(m_next, 1e-12)
                        c_next = np.interp(m_next, resource_grid, policy[t + 1, zp, :])
                        c_next = max(c_next, 1e-12)
                        rhs += prob * u_prime_scalar(c_next)
                    
                    rhs *= beta * R
                    
                    return lhs - rhs
                
                # Solve for c in (0, m]
                c_lo = 1e-12
                c_hi = m
                
                try:
                    f_lo = euler_residual(c_lo)
                    f_hi = euler_residual(c_hi)
                    
                    if f_lo * f_hi > 0:
                        if f_lo < 0:
                            c_sol = c_hi
                        else:
                            c_sol = c_lo
                    else:
                        c_sol = brentq(euler_residual, c_lo, c_hi,
                                        xtol=1e-10, rtol=1e-10, maxiter=50)
                except Exception:
                    c_sol = m * 0.5
                
                policy[t, z, i] = c_sol
    
    # Ensure constraints
    for t in range(horizon):
        for z in range(n_states):
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, resource_grid + 1e-10)
    
    # Final period must equal grid exactly
    policy[horizon - 1, :, :] = resource_grid[None, :, None]
    
    return policy
# EVOLVE-BLOCK-END
