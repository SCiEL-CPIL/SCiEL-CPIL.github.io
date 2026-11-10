import numpy as np
from scipy.interpolate import CubicSpline
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    N = len(resource_grid)
    T = horizon
    Z = len(income)

    # Policy array: c[t, z, i]
    c_policy = np.zeros((T, Z, N))

    # Last period: consume everything
    c_policy[T-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]

    # Value function: V[t, z, m] = max E[sum_{s=t}^{T-1} beta^{s-t} u(c_s)]
    # We store value as a function of m for each z at each t.
    # V[T, z, m] = u(m) for all z (consume everything at T-1)
    # Actually, the problem says c[5] equals grid, so t=5 is last period.
    # horizon = 6, so t = 0,1,2,3,4,5. t=5 is last.
    # V[5, z, m] = u(m)
    # V[t, z, m] = max_c { u(c) + beta * sum_{z'} P(z'|z) V[t+1, z', R(m-c) + income[z']] }

    def utility(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    def util_prime(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return 1.0 / c
        else:
            return c ** (-gamma)

    def util_double_prime(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return -1.0 / c ** 2
        else:
            return -gamma * c ** (-gamma - 1)

    # We'll work in log-space for m to handle the wide range [0.001, 40]
    # Let x = log(m), so m = exp(x)
    # V as function of x for each z at each t

    # Grid for x: use the resource_grid converted to log
    x_grid = np.log(resource_grid)
    N_x = len(x_grid)

    # V[t, z, x_i] for i in 0..N_x-1
    # At t = T-1 = 5: V = u(exp(x))
    V = np.zeros((T, Z, N_x))
    V[T-1, :, :] = utility(resource_grid)[np.newaxis, np.newaxis, :]

    # For t from T-2 down to 0:
    # V[t, z, m] = max_c { u(c) + beta * sum_{z'} P(z'|z) V[t+1, z', R(m-c) + income[z']] }
    # FOC: u'(c) + beta * R * sum_{z'} P(z'|z) V'[t+1, z', R(m-c) + income[z']] = 0
    # where V' is derivative w.r.t. m' = R(m-c) + income[z']

    # We'll use Newton's method on the FOC.
    # For each (t, z, m), solve for c.
    # FOC: u'(c) + beta * R * sum_{z'} P(z'|z) V'[t+1, z', R(m-c) + income[z']] = 0

    # We need V[t+1, z', m'] as a smooth function of m'.
    # Use cubic spline in log-space: V as function of x = log(m)
    # dV/dm = dV/dx * dx/dm = (dV/dx) / m
    # So V'[t+1, z', m'] = spline_x'(x') / m' where x' = log(m')

    # Let's precompute splines for V[t+1, z'] as function of x
    # V_next[z', x] for each z'

    for t in range(T - 2, -1, -1):
        if time.time() > deadline:
            break

        # Build splines for V[t+1, z'] as function of x
        splines = []
        for zp in range(Z):
            spl = CubicSpline(x_grid, V[t+1, zp, :], bc_type='not-a-knot')
            splines.append(spl)

        # For each z and each m (x), solve FOC for c
        for z in range(Z):
            # Transition probabilities from z
            probs = transition[z]

            for i in range(N_x):
                m = resource_grid[i]
                x = x_grid[i]

                # Initial guess: c = m * 0.5 or use previous policy
                # Use a simple heuristic: c is proportional to m
                # For log utility, c ~ m * (1 - beta*R) if income constant
                # Start with c = m * 0.5, bounded
                c = m * 0.5
                c = np.clip(c, 1e-12, m)

                # Newton iterations
                for it in range(20):
                    if time.time() > deadline:
                        break

                    # Compute m' for each z'
                    m_prime = R * (m - c) + income
                    # Ensure m_prime > 0
                    m_prime = np.maximum(m_prime, 1e-12)
                    x_prime = np.log(m_prime)

                    # V'[t+1, z', m'] = dV/dm' = (dV/dx') / m'
                    # dV/dx' from spline derivative
                    dV_dx = np.array([splines[zp](x_prime, 1) for zp in range(Z)])
                    V_prime = dV_dx / m_prime

                    # FOC: u'(c) + beta * R * sum_{z'} P(z'|z) V'[t+1, z', m'] = 0
                    f = util_prime(c) + beta * R * np.dot(probs, V_prime)

                    # Derivative of FOC w.r.t. c
                    # d/dc [u'(c)] = u''(c)
                    # d/dc [beta * R * sum P V'(m')] where m' = R(m-c) + income[z']
                    # dm'/dc = -R
                    # dV'/dc = V''(m') * dm'/dc = -R * V''(m')
                    # V''(m') = d/dm' [dV/dx' / m'] = (d^2V/dx'^2 * (1/m') - dV/dx' / m'^2) * (1/m')
                    # Wait, let's be careful.
                    # V'(m') = dV/dm' = (dV/dx') * (dx'/dm') = (dV/dx') / m'
                    # V''(m') = d/dm' [ (dV/dx') / m' ]
                    # = (d/dm' (dV/dx')) * (1/m') + (dV/dx') * d(1/m')/dm'
                    # d/dm' (dV/dx') = (d^2V/dx'^2) * (dx'/dm') = (d^2V/dx'^2) / m'
                    # d(1/m')/dm' = -1/m'^2
                    # So V''(m') = (d^2V/dx'^2 / m') * (1/m') + (dV/dx') * (-1/m'^2)
                    # = d^2V/dx'^2 / m'^2 - dV/dx' / m'^2
                    # = (d^2V/dx'^2 - dV/dx') / m'^2

                    d2V_dx = np.array([splines[zp](x_prime, 2) for zp in range(Z)])
                    dV_dx_arr = np.array([splines[zp](x_prime, 1) for zp in range(Z)])
                    V_double_prime = (d2V_dx - dV_dx_arr) / m_prime ** 2

                    # df/dc = u''(c) + beta * R * sum P(z'|z) * dV'/dc
                    # dV'/dc = V''(m') * (-R)
                    df_dc = util_double_prime(c) + beta * R * np.dot(probs, V_double_prime) * (-R)

                    if abs(df_dc) < 1e-30:
                        break

                    c_new = c - f / df_dc
                    c_new = np.clip(c_new, 1e-12, m)

                    if abs(c_new - c) < 1e-12:
                        c = c_new
                        break
                    c = c_new

                c_policy[t, z, i] = c

        # Compute V[t, z, x_i] = u(c) + beta * sum P V[t+1, z', m']
        for z in range(Z):
            probs = transition[z]
            for i in range(N_x):
                m = resource_grid[i]
                c = c_policy[t, z, i]
                m_prime = R * (m - c) + income
                m_prime = np.maximum(m_prime, 1e-12)
                x_prime = np.log(m_prime)

                # Interpolate V[t+1, z', x_prime]
                V_next_vals = np.array([splines[zp](x_prime) for zp in range(Z)])
                V[t, z, i] = utility(c) + beta * np.dot(probs, V_next_vals)

    # Check time and return
    if time.time() > deadline:
        # Return what we have
        pass

    return c_policy
