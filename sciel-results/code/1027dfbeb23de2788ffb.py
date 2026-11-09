import numpy as np
from scipy.optimize import brentq
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Grid and states
    m_grid = resource_grid
    n_m = len(m_grid)
    n_z = 2
    T = horizon
    
    # Policy array: [time, state, grid_index]
    c_policy = np.zeros((T, n_z, n_m))
    
    # Utility derivative
    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    # Terminal condition: t = T - 1 (index 5)
    # c[T-1, z, :] = m_grid
    c_policy[T-1, :, :] = m_grid
    
    # Backward induction
    for t in range(T - 2, -1, -1):
        for z in range(n_z):
            for i_m in range(n_m):
                m = m_grid[i_m]
                
                # Define the Euler equation residual
                def euler_residual(c):
                    # c is current consumption
                    # m' = R * (m - c) + income[z']
                    # c_next = c_policy[t+1, z', m'] (interpolated)
                    
                    # Calculate next period resources for each z'
                    m_prime_0 = R * (m - c) + income[0]
                    m_prime_1 = R * (m - c) + income[1]
                    
                    # Interpolate next period consumption
                    # Ensure m_prime is within grid bounds for interpolation
                    # np.interp clamps to endpoints if out of range
                    c_next_0 = np.interp(m_prime_0, m_grid, c_policy[t+1, 0, :])
                    c_next_1 = np.interp(m_prime_1, m_grid, c_policy[t+1, 1, :])
                    
                    # Expected marginal utility of next period consumption
                    eu_next = transition[z, 0] * u_prime(c_next_0) + transition[z, 1] * u_prime(c_next_1)
                    
                    # Current marginal utility
                    mu_current = u_prime(c)
                    
                    # Residual: mu_current - beta * R * E[mu_next]
                    return mu_current - beta * R * eu_next

                # Solve for c in (0, m]
                # We expect a root in (1e-12, m)
                # If no root, check boundaries
                
                c_min = 1e-12
                c_max = m
                
                f_min = euler_residual(c_min)
                f_max = euler_residual(c_max)
                
                if f_min * f_max > 0:
                    # No sign change, use boundary
                    if abs(f_min) < abs(f_max):
                        c_opt = c_min
                    else:
                        c_opt = c_max
                else:
                    # Use Brent's method with tighter tolerance
                    try:
                        c_opt = brentq(euler_residual, c_min, c_max, xtol=1e-12, rtol=1e-12, maxiter=200)
                    except ValueError:
                        # Fallback to boundary if brentq fails
                        if abs(f_min) < abs(f_max):
                            c_opt = c_min
                        else:
                            c_opt = c_max
                
                c_policy[t, z, i_m] = c_opt
                
                # Time check
                if time.time() > deadline:
                    # Fill remaining with current best guess or simple heuristic
                    for t_rem in range(t, -1, -1):
                        for z_rem in range(n_z):
                            if t_rem == t and z_rem == z:
                                for i_rem in range(i_m + 1, n_m):
                                    c_policy[t_rem, z_rem, i_rem] = m_grid[i_rem]
                            else:
                                c_policy[t_rem, z_rem, :] = m_grid
                    return c_policy

    # Final polish: refine each policy point with a tighter local solve
    for t in range(T - 1):
        for z in range(n_z):
            for i_m in range(n_m):
                m = m_grid[i_m]
                c_init = c_policy[t, z, i_m]
                
                def euler_residual_polish(c):
                    m_prime_0 = R * (m - c) + income[0]
                    m_prime_1 = R * (m - c) + income[1]
                    c_next_0 = np.interp(m_prime_0, m_grid, c_policy[t+1, 0, :])
                    c_next_1 = np.interp(m_prime_1, m_grid, c_policy[t+1, 1, :])
                    eu_next = transition[z, 0] * u_prime(c_next_0) + transition[z, 1] * u_prime(c_next_1)
                    mu_current = u_prime(c)
                    return mu_current - beta * R * eu_next
                
                c_min = 1e-12
                c_max = m
                
                f_min = euler_residual_polish(c_min)
                f_max = euler_residual_polish(c_max)
                
                if f_min * f_max > 0:
                    if abs(f_min) < abs(f_max):
                        c_opt = c_min
                    else:
                        c_opt = c_max
                else:
                    try:
                        c_opt = brentq(euler_residual_polish, c_min, c_max, xtol=1e-14, rtol=1e-14, maxiter=300)
                    except ValueError:
                        c_opt = c_init
                
                c_policy[t, z, i_m] = c_opt
                
                if time.time() > deadline:
                    # If time runs out during polish, return what we have (already valid)
                    return c_policy

    return c_policy
