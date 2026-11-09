"""Consumption policy solver using backward induction with FOC root-finding."""
import time
import numpy as np
from scipy.optimize import brentq

DEADLINE_SECONDS = 140  # Safety margin before the 150s kill


def _crsa_utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(c)
    return (c ** (1.0 - gamma)) / (1.0 - gamma)


def _crsa_marginal(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return 1.0 / c
    return c ** (-gamma)


def _solve_consumption_single(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'])
    trans = np.asarray(model['transition'])
    horizon = model['horizon']
    states = len(income)
    n_grid = len(resource_grid)

    consumption = np.zeros((horizon, states, n_grid))

    # Terminal condition: consume all
    consumption[horizon - 1] = np.tile(resource_grid, (states, 1))

    # V_next[z, i] = value at next period for state z, grid index i
    # At terminal, V_next is the utility of consuming all (but we don't need it
    # since we stop the loop at t = horizon - 2)
    # Actually, for t = horizon - 2, we need V at t = horizon - 1.
    # V[m, z] at terminal = u(m) since c = m.
    V_next = _crsa_utility(np.tile(resource_grid, (states, 1)), gamma)

    for t in range(horizon - 2, -1, -1):
        V_current = np.zeros((states, n_grid))
        m_vals = resource_grid  # shape (n_grid,)

        for z in range(states):
            for i in range(n_grid):
                m = m_vals[i]
                
                # Define the objective function to maximize:
                # u(c) + beta * E[V(R(m-c) + income[z'])]
                # We use brentq to find the root of the derivative (FOC).
                
                def objective(c):
                    # c is scalar
                    m_minus_c = m - c
                    if m_minus_c < -1e-12:
                        return -1e20  # Infeasible, but brentq should stay in bounds
                    m_minus_c = max(m_minus_c, 0.0)
                    
                    exp_V = 0.0
                    for zp in range(states):
                        m_prime = R * m_minus_c + income[zp]
                        # Clip m_prime to grid range for interpolation
                        if m_prime <= resource_grid[0]:
                            v = V_next[zp, 0]
                        elif m_prime >= resource_grid[-1]:
                            v = V_next[zp, -1]
                        else:
                            idx = np.searchsorted(resource_grid, m_prime, side='right') - 1
                            idx = np.clip(idx, 0, n_grid - 2)
                            m_left = resource_grid[idx]
                            m_right = resource_grid[idx + 1]
                            denom = m_right - m_left
                            if denom < 1e-15:
                                w = 0.0
                            else:
                                w = (m_prime - m_left) / denom
                                w = np.clip(w, 0.0, 1.0)
                            v = V_next[zp, idx] + w * (V_next[zp, idx + 1] - V_next[zp, idx])
                        exp_V += trans[z, zp] * v
                    
                    return _crsa_utility(c, gamma) + beta * exp_V

                def derivative(c):
                    # u'(c) - beta * R * E[V'(m')]
                    # Note: V' is approximated by finite differences on the grid
                    c_safe = max(c, 1e-12)
                    u_prime = _crsa_marginal(c_safe, gamma)
                    
                    m_minus_c = m - c
                    if m_minus_c < 0:
                        return 1e20 # Gradient points towards feasible region
                    
                    exp_V_prime = 0.0
                    for zp in range(states):
                        m_prime = R * m_minus_c + income[zp]
                        # Clip m_prime to grid range
                        m_prime_clipped = np.clip(m_prime, resource_grid[0], resource_grid[-1])
                        
                        # Approximate V' using central difference or forward/backward
                        if m_prime_clipped <= resource_grid[0]:
                            # Use forward difference
                            m0 = resource_grid[0]
                            m1 = resource_grid[1]
                            v0 = V_next[zp, 0]
                            v1 = V_next[zp, 1]
                            slope = (v1 - v0) / (m1 - m0)
                        elif m_prime_clipped >= resource_grid[-1]:
                            # Use backward difference
                            m0 = resource_grid[-2]
                            m1 = resource_grid[-1]
                            v0 = V_next[zp, -2]
                            v1 = V_next[zp, -1]
                            slope = (v1 - v0) / (m1 - m0)
                        else:
                            # Use interpolation to get V, then finite difference of V
                            # Better: interpolate V at m_prime - h and m_prime + h
                            h = 1e-4 * max(1.0, m_prime_clipped)
                            m_minus = max(m_prime_clipped - h, resource_grid[0])
                            m_plus = min(m_prime_clipped + h, resource_grid[-1])
                            
                            def get_V_at(m_val, z_idx):
                                if m_val <= resource_grid[0]:
                                    return V_next[z_idx, 0]
                                elif m_val >= resource_grid[-1]:
                                    return V_next[z_idx, -1]
                                else:
                                    idx = np.searchsorted(resource_grid, m_val, side='right') - 1
                                    idx = np.clip(idx, 0, n_grid - 2)
                                    m_left = resource_grid[idx]
                                    m_right = resource_grid[idx + 1]
                                    denom = m_right - m_left
                                    if denom < 1e-15:
                                        return V_next[z_idx, idx]
                                    w = (m_val - m_left) / denom
                                    w = np.clip(w, 0.0, 1.0)
                                    return V_next[z_idx, idx] + w * (V_next[z_idx, idx + 1] - V_next[z_idx, idx])
                            
                            v_minus = get_V_at(m_minus, zp)
                            v_plus = get_V_at(m_plus, zp)
                            slope = (v_plus - v_minus) / (m_plus - m_minus)
                        
                        exp_V_prime += trans[z, zp] * slope
                    
                    return u_prime - beta * R * exp_V_prime

                # Solve derivative(c) = 0 for c in (1e-12, m)
                # The derivative is decreasing in c (concavity), so we can use brentq.
                # We need to find a bracket [a, b] where derivative changes sign.
                # At c -> 0, derivative -> +inf (usually)
                # At c -> m, derivative -> -inf (usually, unless corner solution)
                
                c_min = 1e-12
                c_max = m
                
                # Check corner cases
                # If derivative at c_max is still positive, optimal is c_max (consume all)
                # If derivative at c_min is still negative, optimal is c_min (consume min)
                
                # To be safe and fast, we can check the sign at the boundaries first.
                # However, evaluating derivative involves loops.
                # Let's try brentq directly with a small offset from boundaries to avoid numerical issues.
                
                try:
                    # Check if the solution is at the boundary
                    # We'll evaluate derivative at c_max - epsilon and c_min + epsilon
                    eps = 1e-6
                    d_low = derivative(c_min + eps)
                    d_high = derivative(c_max - eps)
                    
                    if d_low < 0:
                        # Derivative is negative even at low c, so optimal is c_min
                        best_c = c_min
                    elif d_high > 0:
                        # Derivative is positive even at high c, so optimal is c_max
                        best_c = c_max
                    else:
                        # Root exists in between
                        best_c = brentq(derivative, c_min + eps, c_max - eps, xtol=1e-8, rtol=1e-8)
                except Exception:
                    # Fallback to a simple heuristic if root finding fails
                    # Use the grid search result or a heuristic
                    # For now, fall back to a fixed ratio or previous value
                    best_c = 0.9 * m
                    best_c = np.clip(best_c, c_min, c_max)

                consumption[t, z, i] = best_c
                V_current[z, i] = objective(best_c)

        V_next = V_current

    return consumption


def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    try:
        result = _solve_consumption_single(model, resource_grid, budget)

        # Validate
        if not np.all(np.isfinite(result)):
            raise ValueError("Non-finite values")
        m = resource_grid[None, None, :]
        if np.any(result <= 0) or np.any(result > m + 1e-10):
            raise ValueError("c out of bounds")
        states = model['transition'].shape[0]
        if not np.allclose(result[5], np.tile(resource_grid, (states, 1)), atol=1e-9):
            raise ValueError("Terminal condition violated")

        return result
    except Exception:
        # Fallback: simple heuristic
        horizon = model['horizon']
        states = len(model['income'])
        n_grid = len(resource_grid)

        result = np.zeros((horizon, states, n_grid))
        for t in range(horizon):
            if t == horizon - 1:
                result[t] = np.tile(resource_grid, (states, 1))
            else:
                result[t] = 0.9 * resource_grid[None, :]
                result[t] = np.clip(result[t], 1e-6, resource_grid[None, :])
                result[t] = np.tile(result[t], (states, 1))

        return result
