"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq

def _utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return np.log(c)
    else:
        return np.power(c, 1 - gamma) / (1 - gamma)

def _utility_prime(c, gamma):
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return 1.0 / c
    else:
        return np.power(c, -gamma)

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = 145.0
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    m_grid = resource_grid  # shape (101,)
    n_m = len(m_grid)
    n_z = 2  # number of income states
    
    c = np.zeros((horizon, n_z, n_m))
    V = np.zeros((horizon + 1, n_z, n_m))
    
    # Terminal condition: at t = horizon-1 (last period), consume everything
    c[horizon - 1, :, :] = m_grid[None, :]
    V[horizon, :, :] = 0.0  # no value after terminal
    
    # For the last period, V[horizon-1] = u(m)
    for z in range(n_z):
        V[horizon - 1, z, :] = _utility(m_grid, gamma)
    
    # For t from horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            # Safety: fill remaining with terminal-like policy
            for tt in range(t, horizon - 1):
                c[tt, :, :] = m_grid[None, :]
                V[tt, :, :] = _utility(m_grid, gamma)
            break
        
        for z in range(n_z):
            for i in range(n_m):
                m = m_grid[i]
                
                # FOC: u'(c) = beta * R * sum_z' P(z'|z) V_{t+1}'(z', m')
                # where m' = R(m - c) + income[z']
                
                def euler_eq(c_val):
                    u_prime = _utility_prime(c_val, gamma)
                    
                    # Compute expected derivative of continuation value
                    cont_deriv = 0.0
                    for zp in range(n_z):
                        m_next = R * (m - c_val) + income[zp]
                        # Need V_{t+1}'(z', m_next)
                        # Use numerical derivative of V[t+1, zp, :]
                        # Clip m_next to grid range
                        m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])
                        
                        # Numerical derivative using central difference
                        eps = 1e-6 * max(1.0, m_next_clipped)
                        m_plus = min(m_next_clipped + eps, m_grid[-1])
                        m_minus = max(m_next_clipped - eps, m_grid[0])
                        
                        V_plus = np.interp(m_plus, m_grid, V[t + 1, zp, :])
                        V_minus = np.interp(m_minus, m_grid, V[t + 1, zp, :])
                        V_deriv = (V_plus - V_minus) / (m_plus - m_minus)
                        
                        cont_deriv += transition[z, zp] * V_deriv
                    
                    return u_prime - beta * R * cont_deriv
                
                # Solve for c in (1e-12, m)
                # Check if solution exists in interior
                eps_lower = 1e-10
                eps_upper = 1e-10
                
                f_lower = euler_eq(eps_lower)
                f_upper = euler_eq(m - eps_upper)
                
                if f_lower < 0:
                    # Optimal c is at lower bound
                    c_opt = eps_lower
                elif f_upper > 0:
                    # Optimal c is at upper bound
                    c_opt = m - eps_upper
                else:
                    # Root exists in interior
                    try:
                        c_opt = brentq(euler_eq, eps_lower, m - eps_upper, xtol=1e-12, rtol=1e-12, maxiter=100)
                    except Exception:
                        # Fallback: bisection
                        lo = eps_lower
                        hi = m - eps_upper
                        for _ in range(60):
                            mid = 0.5 * (lo + hi)
                            if euler_eq(mid) > 0:
                                lo = mid
                            else:
                                hi = mid
                        c_opt = 0.5 * (lo + hi)
                
                c[t, z, i] = c_opt
                # Compute value at optimal c
                u_val = _utility(c_opt, gamma)
                cont_val = 0.0
                for zp in range(n_z):
                    m_next = R * (m - c_opt) + income[zp]
                    m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])
                    V_interp = np.interp(m_next_clipped, m_grid, V[t + 1, zp, :])
                    cont_val += transition[z, zp] * V_interp
                V[t, z, i] = u_val + beta * cont_val
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            c[t, z, :] = np.maximum(c[t, z, :], 1e-12)
            c[t, z, :] = np.minimum(c[t, z, :], m_grid + 1e-10)
    
    # Terminal period: c[5] should equal the grid
    c[horizon - 1, :, :] = m_grid[None, :]
    
    return c
# EVOLVE-BLOCK-END
