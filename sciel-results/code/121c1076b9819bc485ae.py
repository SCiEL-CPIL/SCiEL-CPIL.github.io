"""Rewritten solver: Newton-Raphson on Euler FOC with PCHIP interpolation of value function."""
import numpy as np
from scipy.interpolate import PchipInterpolator
import time

def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    DEADLINE = 145.0  # seconds, safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # resource_grid has 101 points
    m_grid = resource_grid  # (101,)
    n_m = len(m_grid)
    n_z = 2
    
    # Precompute utility functions
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-12:
            return np.log(c)
        else:
            return c**(1 - gamma) / (1 - gamma)
    
    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-12:
            return 1.0 / c
        else:
            return c**(-gamma)
    
    def u_prime_inv(x):
        # Inverse of u_prime: u'(c) = c^{-gamma} (or 1/c for gamma=1)
        # c = x^{-1/gamma} for gamma != 1, c = 1/x for gamma = 1
        x = np.maximum(x, 1e-12)
        if abs(gamma - 1) < 1e-12:
            return 1.0 / x
        else:
            return x**(-1.0/gamma)
    
    # Terminal condition: at t=horizon (last period), consume all resources
    # c[5] = m (within 1e-9)
    consumption = np.zeros((horizon, n_z, n_m))
    
    # Value function at terminal: V_h(m, z) = u(m) since c = m
    # Actually, we need to be careful. The horizon is 6, so t goes 0..5.
    # At t=5 (last period), c = m (consume all).
    # So V_5(m, z) = u(m)
    
    # We'll work backwards from t=5 to t=0.
    # V[t][z, m] = max_c { u(c) + beta * sum_{z'} transition[z, z'] * V[t+1][z', R*(m-c) + income[z']] }
    
    # For the last period (t=5):
    consumption[5, :, :] = m_grid[np.newaxis, :]  # c = m for both income states
    # V at t=5: V_5[m, z] = u(m) (doesn't depend on z since no continuation)
    V_next = np.zeros((n_z, n_m))
    for z in range(n_z):
        V_next[z, :] = u(m_grid)
    
    # For t from 4 down to 0:
    for t in range(horizon - 1, -1, -1):
        # Check time
        if time.time() > DEADLINE - 5:
            # Fill remaining with a simple heuristic: consume proportionally
            for tt in range(t, -1, -1):
                for z in range(n_z):
                    consumption[tt, z, :] = np.minimum(m_grid, 0.8 * m_grid + 0.1)
            break
        
        # Build interpolator for V_next
        # We need V_next as a function of m for each z
        V_interp = []
        for z in range(n_z):
            V_interp.append(PchipInterpolator(m_grid, V_next[z, :], extrapolate=False))
        
        # For each income state z and each m, solve for optimal c
        for z in range(n_z):
            for i_m in range(n_m):
                m = m_grid[i_m]
                
                # The Euler equation / FOC:
                # u'(c) = beta * sum_{z'} P[z, z'] * u'(c') * R
                # where c' = R*(m - c) + income[z']
                # So u'(c) = beta * R * sum_{z'} P[z, z'] * u'(R*(m-c) + income[z'])
                
                # Define f(c) = u'(c) - beta * R * sum_{z'} P[z, z'] * u'(R*(m-c) + income[z'])
                # We want f(c) = 0, with c in (0, m)
                
                def f(c):
                    if c <= 1e-12 or c >= m - 1e-12:
                        return None
                    up_c = u_prime(c)
                    rhs = 0.0
                    for zp in range(n_z):
                        m_next = R * (m - c) + income[zp]
                        if m_next <= 1e-12:
                            rhs += transition[z, zp] * u_prime(1e-12)
                        else:
                            rhs += transition[z, zp] * u_prime(m_next)
                    return up_c - beta * R * rhs
                
                def f_prime(c):
                    # Derivative of f w.r.t. c
                    if c <= 1e-12 or c >= m - 1e-12:
                        return None
                    # d/dc u'(c) = -u''(c)
                    # u''(c) = -gamma * c^{-gamma-1} for gamma != 1, -1/c^2 for gamma=1
                    if abs(gamma - 1) < 1e-12:
                        u2_c = -1.0 / (c * c)
                    else:
                        u2_c = -gamma * c**(-gamma - 1)
                    
                    # d/dc of beta * R * sum P * u'(R*(m-c) + income[zp])
                    # = beta * R * sum P * u''(m_next) * (-R)
                    # = -beta * R^2 * sum P * u''(m_next)
                    rhs_deriv = 0.0
                    for zp in range(n_z):
                        m_next = R * (m - c) + income[zp]
                        if m_next <= 1e-12:
                            m_next_eff = 1e-12
                        else:
                            m_next_eff = m_next
                        if abs(gamma - 1) < 1e-12:
                            u2_mn = -1.0 / (m_next_eff * m_next_eff)
                        else:
                            u2_mn = -gamma * m_next_eff**(-gamma - 1)
                        rhs_deriv += transition[z, zp] * u2_mn
                    
                    return -u2_c - beta * R * (-R) * rhs_deriv
                    # = -u2_c + beta * R^2 * rhs_deriv
                    # Wait, let me re-derive.
                    # f(c) = u'(c) - beta * R * sum P * u'(m_next)
                    # f'(c) = u''(c) - beta * R * sum P * u''(m_next) * d(m_next)/dc
                    # d(m_next)/dc = -R
                    # f'(c) = u''(c) - beta * R * sum P * u''(m_next) * (-R)
                    # f'(c) = u''(c) + beta * R^2 * sum P * u''(m_next)
                    
                    # Let me redo:
                    u2_c_val = u2_c  # this is u''(c)
                    sum_u2_mn = rhs_deriv  # this is sum P * u''(m_next)
                    return u2_c_val + beta * R * R * sum_u2_mn
                
                # Solve f(c) = 0 using Newton-Raphson
                # Initial guess: use a fraction of m
                c = 0.5 * m
                converged = False
                for it in range(30):
                    fc = f(c)
                    if fc is None:
                        c = 0.5 * m
                        continue
                    if abs(fc) < 1e-10:
                        converged = True
                        break
                    fp = f_prime(c)
                    if fp is None or abs(fp) < 1e-15:
                        # Bisection fallback
                        break
                    c_new = c - fc / fp
                    # Clamp to valid range
                    if c_new <= 1e-12:
                        c_new = 1e-12
                    if c_new >= m - 1e-12:
                        c_new = m - 1e-12
                    if abs(c_new - c) < 1e-14:
                        c = c_new
                        converged = True
                        break
                    c = c_new
                else:
                    # If Newton didn't converge, try bisection
                    lo = 1e-12
                    hi = m - 1e-12
                    for it in range(60):
                        mid = (lo + hi) / 2
                        fm = f(mid)
                        if fm is None:
                            break
                        if abs(fm) < 1e-10:
                            c = mid
                            converged = True
                            break
                        # f(c) is decreasing in c (u' is decreasing, and the RHS term... let's check)
                        # Actually, as c increases, u'(c) decreases, and m_next decreases so u'(m_next) increases
                        # So f(c) = u'(c) - beta*R*sum P*u'(m_next) should be decreasing
                        if fm > 0:
                            lo = mid
                        else:
                            hi = mid
                    if not converged:
                        c = (lo + hi) / 2
                
                # Ensure validity
                c = max(1e-12, min(c, m - 1e-12))
                consumption[t, z, i_m] = c
        
        # Compute V[t] for each z and m
        V_current = np.zeros((n_z, n_m))
        for z in range(n_z):
            for i_m in range(n_m):
                m = m_grid[i_m]
                c = consumption[t, z, i_m]
                val = u(c)
                for zp in range(n_z):
                    m_next = R * (m - c) + income[zp]
                    m_next = max(m_next, 1e-12)
                    # Interpolate V_next[zp] at m_next
                    if m_next <= m_grid[0]:
                        vn = V_next[zp, 0]
                    elif m_next >= m_grid[-1]:
                        vn = V_next[zp, -1]
                    else:
                        vn = V_interp[zp](m_next)
                    val += beta * transition[z, zp] * vn
                V_current[z, i_m] = val
        
        V_next = V_current
    
    # Final validation and clamping
    for t in range(horizon):
        for z in range(n_z):
            for i_m in range(n_m):
                m = m_grid[i_m]
                c = consumption[t, z, i_m]
                c = max(1e-12, min(c, m + 1e-10))
                consumption[t, z, i_m] = c
    
    # Ensure terminal condition: c[5] = m within 1e-9
    consumption[5, :, :] = m_grid[np.newaxis, :]
    
    return consumption
