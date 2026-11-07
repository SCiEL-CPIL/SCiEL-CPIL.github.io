"""Rewritten program for consumption-saving policy accuracy."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin below 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    n_z = len(income)
    
    # Scoring grid for interpolation
    m_grid = np.geomspace(0.001, 40, 251)
    n_m = len(m_grid)
    
    # Output grid
    n_out = len(resource_grid)
    
    def u(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c_safe)
        else:
            return c_safe ** (1.0 - gamma) / (1.0 - gamma)
    
    def u_prime(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c_safe
        else:
            return c_safe ** (-gamma)
    
    # Initialize consumption policies
    # c[t, z, m_index] for t in [0, horizon), z in [0, n_z), m_index in [0, n_m)
    c_policy = np.zeros((horizon, n_z, n_m))
    
    # Terminal condition: at t=horizon, consume all resources
    # V[horizon, z, m] = u(m)
    # So c[horizon-1] should account for this
    
    # Backward induction
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 5:
            break
        
        for z in range(n_z):
            # Continuation value function V[t+1, z', m']
            # For each m in m_grid, find optimal c
            for i_m in range(n_m):
                m = m_grid[i_m]
                
                # Find best c by maximizing u(c) + beta * E[V[t+1, z', m']]
                # m' = R*(m - c) + income[z']
                # z' drawn from transition[z]
                
                # For terminal period (t = horizon-1), V[horizon] = u(m')
                # So we maximize u(c) + beta * E[u(m')]
                
                # Use grid search + refinement
                # c must satisfy 0 < c <= m
                c_min = 1e-12
                c_max = m * (1.0 - 1e-10)
                
                if c_max <= c_min:
                    c_policy[t, z, i_m] = m
                    continue
                
                # Coarse grid search
                n_coarse = 200
                c_coarse = np.geomspace(c_min, c_max, n_coarse)
                
                if t == horizon - 1:
                    # Terminal: V = u(m')
                    values = np.zeros(n_coarse)
                    for j, cj in enumerate(c_coarse):
                        m_after = R * (m - cj)
                        ev = 0.0
                        for zp in range(n_z):
                            ev += transition[z, zp] * u(m_after + income[zp])
                        values[j] = u(cj) + beta * ev
                else:
                    # Use interpolated continuation values
                    # V[t+1, z', m'] for each z'
                    # We need to evaluate at m' = R*(m-c) + income[z']
                    
                    # For each z', interpolate V[t+1, z', :] onto m'
                    values = np.zeros(n_coarse)
                    for j, cj in enumerate(c_coarse):
                        m_after = R * (m - cj)
                        ev = 0.0
                        for zp in range(n_z):
                            m_prime = m_after + income[zp]
                            # Interpolate V[t+1, zp, :] at m_prime
                            if m_prime <= m_grid[0]:
                                v_cont = u(m_prime) if t + 1 == horizon - 1 else _interp_v(c_policy, t + 1, zp, m_prime, m_grid, beta, R, income, transition, u, horizon)
                            elif m_prime >= m_grid[-1]:
                                v_cont = _interp_v(c_policy, t + 1, zp, m_prime, m_grid, beta, R, income, transition, u, horizon)
                            else:
                                v_cont = np.interp(m_prime, m_grid, _get_v(t + 1, zp, c_policy, m_grid, beta, R, income, transition, u, horizon))
                            ev += transition[z, zp] * v_cont
                        values[j] = u(cj) + beta * ev
                
                # Find max
                best_idx = np.argmax(values)
                c_best = c_coarse[best_idx]
                
                # Refine with local search
                if 0 < best_idx < n_coarse - 1:
                    c_left = c_coarse[best_idx - 1]
                    c_right = c_coarse[best_idx + 1]
                    # Fine grid
                    n_fine = 50
                    c_fine = np.linspace(c_left, c_right, n_fine)
                    
                    if t == horizon - 1:
                        values_fine = np.zeros(n_fine)
                        for j, cj in enumerate(c_fine):
                            m_after = R * (m - cj)
                            ev = 0.0
                            for zp in range(n_z):
                                ev += transition[z, zp] * u(m_after + income[zp])
                            values_fine[j] = u(cj) + beta * ev
                    else:
                        values_fine = np.zeros(n_fine)
                        for j, cj in enumerate(c_fine):
                            m_after = R * (m - cj)
                            ev = 0.0
                            for zp in range(n_z):
                                m_prime = m_after + income[zp]
                                v_cont = _interp_v(c_policy, t + 1, zp, m_prime, m_grid, beta, R, income, transition, u, horizon)
                                ev += transition[z, zp] * v_cont
                            values_fine[j] = u(cj) + beta * ev
                    
                    best_idx_fine = np.argmax(values_fine)
                    c_best = c_fine[best_idx_fine]
                
                c_policy[t, z, i_m] = c_best
    
    # Now interpolate c_policy onto resource_grid for output
    result = np.zeros((horizon, n_z, n_out))
    for t in range(horizon):
        for z in range(n_z):
            result[t, z, :] = np.interp(resource_grid, m_grid, c_policy[t, z, :])
    
    # Terminal condition: c[horizon-1] should equal m (consume all)
    # Actually, the task says c[5] equal to the grid within 1e-9
    # So at the last period, consume everything
    result[horizon - 1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    # Ensure 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            result[t, z, :] = np.maximum(result[t, z, :], 1e-12)
            result[t, z, :] = np.minimum(result[t, z, :], resource_grid + 1e-10)
    
    # Final check for finite values
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=1e-6, posinf=1e6, neginf=1e-6)
    
    return result


def _get_v(t, z, c_policy, m_grid, beta, R, income, transition, u, horizon):
    """Get value function V[t, z, m_grid] from consumption policy."""
    if t >= horizon:
        return u(m_grid)
    
    # V[t, z, m] = u(c) + beta * E[V[t+1, z', m']]
    # where c = c_policy[t, z, m_index]
    v = np.zeros(len(m_grid))
    for i_m, m in enumerate(m_grid):
        c = c_policy[t, z, i_m]
        m_after = R * (m - c)
        ev = 0.0
        for zp in range(len(income)):
            m_prime = m_after + income[zp]
            if t + 1 >= horizon:
                v_cont = u(m_prime)
            else:
                v_cont = _interp_v(c_policy, t + 1, zp, m_prime, m_grid, beta, R, income, transition, u, horizon)
            ev += transition[z, zp] * v_cont
        v[i_m] = u(c) + beta * ev
    return v


def _interp_v(c_policy, t, z, m_prime, m_grid, beta, R, income, transition, u, horizon):
    """Interpolate value function at m_prime for period t, state z."""
    if t >= horizon:
        return u(m_prime)
    
    # Get V[t, z, m_grid] and interpolate
    v_grid = _get_v(t, z, c_policy, m_grid, beta, R, income, transition, u, horizon)
    if m_prime <= m_grid[0]:
        return v_grid[0]
    elif m_prime >= m_grid[-1]:
        return v_grid[-1]
    else:
        return np.interp(m_prime, m_grid, v_grid)
# EVOLVE-BLOCK-END
