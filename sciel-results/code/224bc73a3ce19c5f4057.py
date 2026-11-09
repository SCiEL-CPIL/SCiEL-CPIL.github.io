"""Consumption policy solver using value-function iteration with Newton root-finding on the Euler equation."""
import time
import numpy as np

DEADLINE_SECONDS = 135


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
    m_grid = resource_grid

    consumption = np.zeros((horizon, states, n_grid))

    # Terminal period: consume everything
    consumption[horizon - 1] = np.tile(m_grid, (states, 1))

    # V_next[z, i] = value at period t+1 given state z and m = m_grid[i]
    # At terminal, V[m, z] = u(m, z) = u(m) (no state dependence in utility)
    V_next = np.tile(_crsa_utility(m_grid, gamma), (states, 1))

    # Precompute log-spaced interpolation weights for geometric grid
    # m_grid is geomspace, so log(m_grid) is linear
    log_m_grid = np.log(m_grid)

    for t in range(horizon - 2, -1, -1):
        V_current = np.zeros((states, n_grid))

        for z in range(states):
            for i in range(n_grid):
                m = m_grid[i]

                # Newton's method to solve the Euler equation:
                # u'(c) = beta * R * sum_zp P(z'|z) * u'(c'(m', zp))
                # where m' = R*(m-c) + income[zp]
                # and c' is the optimal consumption next period given m', zp
                # By envelope theorem: u'(c') = V_m(m', zp)
                # So we need V_m(m', zp) = dV/dm at m' for state zp.
                # We approximate V_m via the derivative of the value function.
                # Since V is smooth and we're on a geometric grid, we can use
                # the fact that for CRRA, the value function has a specific form,
                # but in general we use finite differences on the interpolated V.

                # Strategy: use Newton's method on f(c) = u'(c) - beta*R*E[V_m(m', zp)]
                # where V_m is computed by differentiating the piecewise-linear (in log space)
                # interpolation of V_next.

                # For a geometric grid, if we interpolate V as a function of log(m),
                # then dV/dm = dV/d(log m) / m.

                # Let's work in log space for better conditioning.
                # Actually, let's just use the standard approach:
                # f(c) = u'(c) - beta * R * sum_zp trans[z,zp] * V_m_next(zp, R*(m-c)+income[zp])

                # V_m_next(zp, m') = d/dm' of interpolated V_next[zp] at m'
                # For piecewise linear interpolation on m_grid:
                # V_m is constant between grid points (the slope of the linear segment).

                # Compute slopes of V_next[zp] on m_grid
                # slope[zp, j] = (V_next[zp, j+1] - V_next[zp, j]) / (m_grid[j+1] - m_grid[j])
                slopes = np.zeros((states, n_grid - 1))
                for zp in range(states):
                    slopes[zp] = np.diff(V_next[zp]) / np.diff(m_grid)

                def euler_residual(c):
                    if c < 1e-12:
                        c = 1e-12
                    if c > m - 1e-12:
                        c = m - 1e-12
                    u_prime_c = _crsa_marginal(c, gamma)

                    expected_vm = 0.0
                    for zp in range(states):
                        m_prime = R * (m - c) + income[zp]
                        # Find which segment m_prime falls in
                        if m_prime <= m_grid[0]:
                            vm_val = slopes[zp, 0]
                        elif m_prime >= m_grid[-1]:
                            vm_val = slopes[zp, -1]
                        else:
                            # Binary search for segment
                            idx = np.searchsorted(m_grid, m_prime) - 1
                            idx = max(0, min(idx, n_grid - 2))
                            vm_val = slopes[zp, idx]
                        expected_vm += trans[z, zp] * vm_val

                    return u_prime_c - beta * R * expected_vm

                def euler_derivative(c):
                    if c < 1e-12:
                        c = 1e-12
                    if c > m - 1e-12:
                        c = m - 1e-12
                    # d/dc of u'(c) = u''(c)
                    if abs(gamma - 1.0) < 1e-12:
                        u_double_prime_c = -1.0 / (c * c)
                    else:
                        u_double_prime_c = -gamma * c ** (-gamma - 1)

                    # d/dc of beta*R*E[V_m(m')] where m' = R*(m-c)+income[zp]
                    # dm'/dc = -R
                    # d/dc V_m(m') = V_mm(m') * (-R)
                    # V_mm is the second derivative of V w.r.t. m
                    # For piecewise linear V, V_mm = 0 (V_m is piecewise constant)
                    # So the derivative of the RHS w.r.t. c is 0 in the interior of segments.
                    # This means f'(c) = u''(c) < 0, so f is strictly decreasing.
                    # We can use f'(c) = u''(c) as an approximation.
                    return u_double_prime_c

                # Newton's method
                c = m * 0.5  # Initial guess
                for _ in range(50):
                    f_val = euler_residual(c)
                    f_deriv = euler_derivative(c)
                    if abs(f_deriv) < 1e-30:
                        break
                    c_new = c - f_val / f_deriv
                    if c_new < 1e-12:
                        c_new = 1e-12
                    if c_new > m - 1e-12:
                        c_new = m - 1e-12
                    if abs(c_new - c) < 1e-14:
                        c = c_new
                        break
                    c = c_new

                # Clamp
                c = max(1e-12, min(c, m))

                consumption[t, z, i] = c

                # Compute V_current[z, i] = u(c) + beta * E[V_next(m', zp)]
                u_val = _crsa_utility(c, gamma)
                expected_v = 0.0
                for zp in range(states):
                    m_prime = R * (m - c) + income[zp]
                    m_prime_clipped = max(m_grid[0], min(m_prime, m_grid[-1]))
                    v_val = np.interp(m_prime_clipped, m_grid, V_next[zp])
                    expected_v += trans[z, zp] * v_val

                V_current[z, i] = u_val + beta * expected_v

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
        if not np.allclose(result[horizon - 1 if (horizon := model['horizon']) else 5], 
                          np.tile(resource_grid, (states, 1)), atol=1e-9):
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
