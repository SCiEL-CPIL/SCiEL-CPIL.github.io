"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    grid = np.asarray(resource_grid, dtype=float)

    n_states = len(income)
    n_grid = len(grid)

    # CRRA utility and marginal utility
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1.0 - gamma)) / (1.0 - gamma)

    def marginal_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)

    def value_derivative(m, z, t_plus_1):
        """Compute V'(m, z, t+1) via central differences with one-sided at boundaries."""
        if m <= grid[0]:
            # Forward difference
            dm = grid[1] - grid[0]
            return (V[t_plus_1, z, 1] - V[t_plus_1, z, 0]) / dm
        elif m >= grid[-1]:
            # Backward difference
            dm = grid[-1] - grid[-2]
            return (V[t_plus_1, z, -1] - V[t_plus_1, z, -2]) / dm
        else:
            # Central difference using nearest grid points
            idx = np.searchsorted(grid, m)
            if idx == 0:
                dm = grid[1] - grid[0]
                return (V[t_plus_1, z, 1] - V[t_plus_1, z, 0]) / dm
            elif idx >= n_grid:
                dm = grid[-1] - grid[-2]
                return (V[t_plus_1, z, -1] - V[t_plus_1, z, -2]) / dm
            else:
                # Use points on either side
                if idx < n_grid:
                    left_m = grid[idx - 1]
                    right_m = grid[idx]
                    left_v = V[t_plus_1, z, idx - 1]
                    right_v = V[t_plus_1, z, idx]
                    dm = right_m - left_m
                    if dm > 0:
                        return (right_v - left_v) / dm
                    else:
                        return 0.0
                else:
                    dm = grid[-1] - grid[-2]
                    return (V[t_plus_1, z, -1] - V[t_plus_1, z, -2]) / dm

    # Value function V[t, z, m] stored on the resource grid
    # Terminal: consume all
    V = np.zeros((horizon, n_states, n_grid))
    C = np.zeros((horizon, n_states, n_grid))

    # Last period: c = m
    V[horizon - 1, :, :] = utility(grid)
    C[horizon - 1, :, :] = grid

    # Backward induction using FOC-based root finding
    # Euler equation: u'(c) = beta * R * sum_{z'} P(z'|z) * V'(R*(m-c) + income[z'], z', t+1)
    
    for t in range(horizon - 2, -1, -1):
        elapsed = time.time() - start_time
        if elapsed > deadline - 3.0:
            # Fill remaining periods with a simple heuristic
            for t2 in range(t, -1, -1):
                for z in range(n_states):
                    C[t2, z, :] = np.clip(grid * 0.4, 1e-12, grid)
                    V[t2, z, :] = utility(C[t2, z, :])
            break

        for z in range(n_states):
            for i_m in range(n_grid):
                m = grid[i_m]
                
                if time.time() > deadline - 1.0:
                    C[t, z, i_m] = np.clip(m * 0.4, 1e-12, m)
                    V[t, z, i_m] = utility(C[t, z, i_m])
                    continue

                # Define the FOC residual: u'(c) - beta * R * E[V'(m')]
                # where m' = R*(m-c) + income[z']
                def foc_residual(c):
                    c = max(c, 1e-12)
                    mu = marginal_utility(np.array([c]))[0]
                    
                    expected_vprime = 0.0
                    for zp in range(n_states):
                        m_prime = R * (m - c) + income[zp]
                        m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
                        vp = value_derivative(m_prime_clipped, zp, t + 1)
                        expected_vprime += transition[z, zp] * vp
                    
                    return mu - beta * R * expected_vprime

                # Try to find root using Brent's method
                # Search range: c in [1e-12, m]
                c_low = 1e-12
                c_high = m
                
                try:
                    f_low = foc_residual(c_low)
                    f_high = foc_residual(c_high)
                    
                    if f_low * f_high < 0:
                        c_opt = brentq(foc_residual, c_low, c_high, xtol=1e-10, maxiter=50)
                    else:
                        # No sign change - use grid search fallback
                        f_coarse = np.linspace(0.001, 1.0, 50)
                        best_f = f_coarse[0]
                        best_res = abs(foc_residual(f_coarse[0] * m))
                        
                        for f in f_coarse[1:]:
                            c = f * m
                            res = abs(foc_residual(c))
                            if res < best_res:
                                best_res = res
                                best_f = f
                        
                        c_opt = best_f * m
                except:
                    # Fallback to simple heuristic
                    c_opt = m * 0.3

                c_opt = np.clip(c_opt, 1e-12, m + 1e-10)
                C[t, z, i_m] = c_opt
                
                # Compute value
                expected_cont = 0.0
                for zp in range(n_states):
                    m_prime = R * (m - c_opt) + income[zp]
                    m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
                    cont_val = np.interp(m_prime_clipped, grid, V[t + 1, zp, :])
                    expected_cont += transition[z, zp] * cont_val
                
                V[t, z, i_m] = utility(np.array([c_opt]))[0] + beta * expected_cont

    # Ensure validity
    C = np.clip(C, 1e-12, None)
    for t in range(horizon):
        for z in range(n_states):
            C[t, z, :] = np.minimum(C[t, z, :], grid + 1e-10)

    # Terminal condition
    C[horizon - 1, :, :] = grid.copy()

    return C
# EVOLVE-BLOCK-END
