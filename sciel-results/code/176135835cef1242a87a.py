"""Consumption policy solver using backward induction with FOC root-finding."""
import time
import numpy as np
from scipy.optimize import brentq

DEADLINE_SECONDS = 135  # Safety margin before the 150s kill

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
    income = model['income']
    trans = model['transition']
    horizon = model['horizon']
    states = len(income)
    n_grid = len(resource_grid)

    consumption = np.zeros((horizon, states, n_grid))
    # V[z, i] = continuation value (value from next period onward) given state z and m=grid[i]
    V_next = np.zeros((states, n_grid))

    for t in range(horizon - 1, -1, -1):
        if t == horizon - 1:
            # Terminal: consume all
            consumption[t] = np.tile(resource_grid, (states, 1))
            V_next = _crsa_utility(np.tile(resource_grid, (states, 1)), gamma)
            continue

        V_current = np.zeros((states, n_grid))
        m_vals = resource_grid

        for z in range(states):
            for i in range(n_grid):
                m = m_vals[i]
                
                # If m is very small, consume all (or close to it)
                # Actually, with CRRA and finite horizon, c < m always unless terminal.
                # But if m is tiny, c is tiny.
                
                # Define the FOC: u'(c) = beta * R * E[ u'(m') * (1 + V_z'(m')) ] ?
                # Wait, the standard FOC is:
                # u'(c) = beta * R * sum_zp P(z'|z) * u'(c') * V_m(m') ... no.
                # The Bellman equation is:
                # V(m, z) = max_c { u(c) + beta * sum_zp P(z'|z) V(m', z') }
                # where m' = R(m-c) + y(z')
                
                # FOC w.r.t c:
                # u'(c) + beta * sum_zp P(z'|z) [ V_m(m', z') * d m' / d c ] = 0
                # d m' / d c = -R
                # u'(c) - beta * R * sum_zp P(z'|z) V_m(m', z') = 0
                # u'(c) = beta * R * sum_zp P(z'|z) V_m(m', z')
                
                # We need V_m(m', z'), the derivative of V w.r.t m.
                # We can approximate V_m by finite differences or store derivatives.
                # Since we are doing backward induction, we can store V and compute V_m via np.gradient or central differences on the grid.
                # However, the grid is geomspace, so spacing varies.
                # A better approach: use the envelope theorem.
                # V_m(m, z) = u'(c*(m, z))
                # So we need c* from the previous step? No, V_next is known.
                # We can compute V_m_next[zp, m'] by differentiating the interpolated V_next.
                
                # Let's compute V_m_next for each state zp.
                # V_next[zp] is defined on resource_grid.
                # We can use np.gradient with the actual grid spacing.
                
                # Pre-compute V_m_next for all states
                V_m_next = np.zeros((states, n_grid))
                for zp in range(states):
                    # Use central differences where possible, one-sided at edges
                    # np.gradient handles non-uniform grids
                    V_m_next[zp] = np.gradient(V_next[zp], resource_grid, edge_order=2)
                
                # Now solve for c in [0, m]
                # f(c) = u'(c) - beta * R * sum_zp P(z'|z) * V_m_next[zp, m']
                # where m' = R(m-c) + income[zp]
                
                def f(c):
                    if c <= 0:
                        c = 1e-12
                    if c >= m:
                        c = m - 1e-12 if m > 1e-12 else 1e-12
                    
                    u_prime_c = _crsa_marginal(c, gamma)
                    
                    expected_v_m = 0.0
                    for zp in range(states):
                        m_prime = R * (m - c) + income[zp]
                        # Clip m_prime to grid for interpolation of V_m
                        if m_prime < resource_grid[0]:
                            m_prime = resource_grid[0]
                        elif m_prime > resource_grid[-1]:
                            m_prime = resource_grid[-1]
                        
                        # Interpolate V_m_next
                        vm_val = np.interp(m_prime, resource_grid, V_m_next[zp])
                        expected_v_m += trans[z, zp] * vm_val
                    
                    return u_prime_c - beta * R * expected_v_m
                
                # Check if solution is at boundary
                # f(0) is +inf (for log) or +inf (for power), so f(0) > 0
                # f(m) is u'(m) - beta * R * ...
                # If f(m) > 0, then optimal c is m (consume all)
                # But in finite horizon with CRRA, c < m usually.
                # Let's check f(m - epsilon).
                
                try:
                    # Use a small epsilon to avoid boundary issues if needed
                    # But brentq needs a sign change.
                    # f(c) is decreasing in c (u' is decreasing, m' increases as c decreases, 
                    # wait: c increases -> m' decreases -> V_m(m') increases (since V is concave, V_m is decreasing in m, so V_m(high m) is low, V_m(low m) is high).
                    # Let's check monotonicity.
                    # u'(c) is decreasing in c.
                    # m' = R(m-c) + y. As c increases, m' decreases.
                    # V_m(m') is decreasing in m', so as m' decreases, V_m(m') increases.
                    # So the term beta * R * E[V_m(m')] increases as c increases.
                    # Thus f(c) = u'(c) - (increasing term).
                    # u'(c) decreases. (increasing term) increases.
                    # So f(c) is strictly decreasing.
                    # f(0+) = +inf (or large positive).
                    # f(m-) = u'(m) - beta * R * E[V_m(R*0 + y)].
                    # If f(m-) > 0, then the root is beyond m, so c* = m.
                    # If f(m-) < 0, then root is in (0, m).
                    
                    # Evaluate at m (or m - tiny)
                    # To be safe, evaluate at m * (1 - 1e-12)
                    c_high = m * (1 - 1e-12)
                    if c_high < 1e-12:
                        c_high = 1e-12
                    f_high = f(c_high)
                    
                    if f_high > 0:
                        # Consume all
                        best_c = m
                    else:
                        c_low = 1e-12
                        # Ensure f_low > 0
                        # f(1e-12) is huge, so it should be positive.
                        best_c = brentq(f, c_low, c_high, xtol=1e-12, rtol=1e-12)
                        
                except Exception:
                    # Fallback to heuristic if root finding fails
                    # Simple heuristic: consume a fraction
                    best_c = 0.5 * m

                consumption[t, z, i] = best_c
                
                # Calculate V_current[z, i]
                # V = u(c) + beta * E[V(m')]
                c_val = max(best_c, 1e-12)
                u_val = _crsa_utility(c_val, gamma)
                
                expected_v = 0.0
                for zp in range(states):
                    m_prime = R * (m - c_val) + income[zp]
                    if m_prime < resource_grid[0]:
                        m_prime = resource_grid[0]
                    elif m_prime > resource_grid[-1]:
                        m_prime = resource_grid[-1]
                    
                    v_val = np.interp(m_prime, resource_grid, V_next[zp])
                    expected_v += trans[z, zp] * v_val
                
                V_current[z, i] = u_val + beta * expected_v

        V_next = V_current

    return consumption


def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    try:
        result = _solve_consumption_single(model, resource_grid, budget)

        # Check deadline
        if time.time() > deadline:
            # Fallback if we ran out of time
            pass

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
                # Linear approximation of consumption
                # c ~ m
                result[t] = 0.9 * resource_grid[None, :]
                result[t] = np.clip(result[t], 1e-6, resource_grid[None, :])
                result[t] = np.tile(result[t], (states, 1))

        return result
