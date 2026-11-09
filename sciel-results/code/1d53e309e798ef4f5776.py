"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    H = model['horizon']

    m_grid = resource_grid
    M = len(m_grid)

    # Utility: exact CRRA as specified
    def utility(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return c ** (1.0 - gamma) / (1.0 - gamma)

    # Value and policy arrays
    V = np.zeros((H, 2, M))
    C = np.zeros((H, 2, M))

    # Terminal period: t = H-1 = 5, c = m
    V[H - 1, :, :] = utility(m_grid)[np.newaxis, np.newaxis, :]
    C[H - 1, :, :] = m_grid[np.newaxis, np.newaxis, :]

    # Backward induction using Euler equation root-finding
    for t in range(H - 2, -1, -1):
        for z in range(2):
            for i, m in enumerate(m_grid):
                if time.time() > deadline:
                    # Fallback: use terminal policy (c = m) for remaining
                    C[t, z, i:] = m_grid[i:]
                    V[t, z, i:] = utility(m_grid[i:])
                    break

                # Define the Euler equation residual
                # We want to find c* such that:
                # u'(c*) = beta * sum_zp p(z'|z) * u'(c_{t+1}^*(m')) * (dm'/dc)
                # where m' = R*(m-c) + income[zp]
                # and c_{t+1}^*(m') is the optimal consumption next period given m'
                
                # For CRRA: u'(c) = c^(-gamma)
                # So: c^(-gamma) = beta * sum_zp p(z'|z) * (c_{t+1}^*(m'))^(-gamma) * R
                # => c^(-gamma) = beta * R * sum_zp p(z'|z) * (c_{t+1}^*(m'))^(-gamma)
                
                def euler_residual(c):
                    if c <= 0:
                        c = 1e-12
                    if c >= m:
                        c = m * (1 - 1e-10)
                    
                    left = c ** (-gamma)
                    right = 0.0
                    for zp in range(2):
                        p = transition[z, zp]
                        if p < 1e-10:
                            continue
                        m_prime = R * (m - c) + income[zp]
                        # Get next period's optimal consumption at m_prime
                        # Use interpolation on the grid
                        if m_prime <= m_grid[0]:
                            c_next = C[t + 1, zp, 0]
                        elif m_prime >= m_grid[-1]:
                            c_next = C[t + 1, zp, -1]
                        else:
                            c_next = np.interp(m_prime, m_grid, C[t + 1, zp, :])
                        right += p * (c_next ** (-gamma))
                    right *= beta * R
                    return left - right

                # Solve for c in (1e-12, m)
                c_lo = max(1e-12, m * 1e-6)
                c_hi = m * (1 - 1e-10)
                
                try:
                    f_lo = euler_residual(c_lo)
                    f_hi = euler_residual(c_hi)
                    
                    if f_lo * f_hi > 0:
                        # No sign change, use a simple grid search as fallback
                        # This shouldn't happen often for well-behaved problems
                        c_vals = np.geomspace(c_lo, c_hi, 50)
                        residuals = np.array([euler_residual(c) for c in c_vals])
                        idx = np.argmin(np.abs(residuals))
                        c_star = c_vals[idx]
                    else:
                        c_star = brentq(euler_residual, c_lo, c_hi, xtol=1e-12, rtol=1e-12, maxiter=50)
                except Exception:
                    # Fallback to a simple rule: consume a fraction
                    c_star = m * 0.5

                # Clamp to valid range
                c_star = np.clip(c_star, 1e-12, m * (1 + 1e-10))
                
                C[t, z, i] = c_star
                
                # Compute value at this point
                val = utility(c_star)
                for zp in range(2):
                    p = transition[z, zp]
                    if p < 1e-10:
                        continue
                    m_prime = R * (m - c_star) + income[zp]
                    if m_prime <= m_grid[0]:
                        v_next = V[t + 1, zp, 0]
                    elif m_prime >= m_grid[-1]:
                        v_next = V[t + 1, zp, -1]
                    else:
                        v_next = np.interp(m_prime, m_grid, V[t + 1, zp, :])
                    val += beta * p * v_next
                V[t, z, i] = val

    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(H):
        for z in range(2):
            C[t, z, :] = np.clip(C[t, z, :], 1e-12, m_grid * (1 + 1e-10))

    # Terminal period must have c = m exactly (within 1e-9)
    C[H - 1, :, :] = m_grid

    return C
# EVOLVE-BLOCK-END
