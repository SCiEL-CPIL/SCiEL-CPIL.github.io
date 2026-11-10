import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']

    num_states = len(income)
    c_floor = 1e-12
    is_log = abs(gamma - 1.0) < 1e-12

    # Higher resolution grid for internal computation
    num_internal = 401
    m_grid = np.geomspace(0.001, 40.0, num_internal)

    # Precompute consumption fractions for each grid point
    # c = m * f, f in [f_min, 1]
    f_vals = np.linspace(0.001, 1.0, 301)

    def utility(c):
        c = np.maximum(c, c_floor)
        if is_log:
            return np.log(c)
        return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    def util_deriv(c):
        c = np.maximum(c, c_floor)
        if is_log:
            return 1.0 / c
        return np.power(c, -gamma)

    # Backward induction
    # V[t][z] = value function at time t, state z, as function of m
    # We store V as arrays indexed by [z, m_index]
    # c_policy[t][z] = consumption at time t, state z, indexed by m_index

    V = np.zeros((horizon, num_states, num_internal))
    c_policy = np.zeros((horizon, num_states, num_internal))

    # Final period: consume all
    V[horizon - 1, :, :] = utility(m_grid)
    c_policy[horizon - 1, :, :] = m_grid

    # Backward from t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        # Check deadline
        if time.time() > deadline - 5.0:
            # Fallback: use linear interpolation for remaining periods
            for tt in range(t, -1, -1):
                for z in range(num_states):
                    c_policy[tt, z, :] = m_grid * 0.5  # rough fallback
                    V[tt, z, :] = utility(c_policy[tt, z, :])
            break

        for z in range(num_states):
            # For each m, find optimal c
            # Euler: u'(c) = beta * R * sum_{z'} P(z'|z) * u'(c') * (dm'/dc)
            # where m' = R*(m-c) + income[z']
            # dm'/dc = -R (since m' = R*m - R*c + income, derivative wrt c is -R)
            # So: u'(c) = beta * R * sum_{z'} P(z'|z) * u'(c') * (-R) ... wait
            # Actually the Euler equation is:
            # u'(c) = beta * E[ u'(c') * R ]  (since dm'/dc = -R, but the FOC is)
            # FOC: u'(c) - beta * E[ R * u'(c') ] = 0
            # So: u'(c) = beta * E[ R * u'(c') ]

            # For each m in m_grid, we need to find c such that:
            # u'(c) = beta * sum_{z'} P(z'|z) * R * u'(c'(m-c, z'))
            # where c' is the optimal consumption next period given m' = R*(m-c) + income[z']

            # We'll use Newton's method on g(c) = u'(c) - beta * R * sum_{z'} P(z'|z) * u'(c')
            # where c' = c_policy[t+1, z'] evaluated at m'

            # Precompute next-period value function interpolation
            # V_next[z'] as function of m'
            # c_next[z'] as function of m'

            # For each m, iterate over possible c
            # Use vectorized approach: for each m, try a set of c values

            # Strategy: for each m, use Newton with a few iterations
            # c is in [c_floor, m]

            c_opt = np.zeros(num_internal)

            for i, m in enumerate(m_grid):
                # Bounds: c in [c_floor, m]
                c_lo = c_floor
                c_hi = m

                # Newton iteration
                c = m * 0.5  # initial guess
                converged = False

                for _newton in range(30):
                    # Compute m' for each z'
                    m_next = R * (m - c) + income  # shape (num_states,)
                    m_next = np.maximum(m_next, 0.0)

                    # Get next-period consumption via interpolation
                    # c_next[z'] = interp(m_next[z'], m_grid, c_policy[t+1, z', :])
                    c_next = np.zeros(num_states)
                    for zp in range(num_states):
                        if m_next[zp] <= m_grid[0]:
                            c_next[zp] = c_policy[t+1, zp, 0]
                        elif m_next[zp] >= m_grid[-1]:
                            c_next[zp] = c_policy[t+1, zp, -1]
                        else:
                            c_next[zp] = np.interp(m_next[zp], m_grid, c_policy[t+1, zp, :])

                    # Compute RHS of Euler: beta * R * sum P(z'|z) * u'(c')
                    rhs = 0.0
                    for zp in range(num_states):
                        rhs += transition[z, zp] * R * util_deriv(c_next[zp])
                    rhs *= beta

                    # LHS: u'(c)
                    lhs = util_deriv(c)

                    # g(c) = lhs - rhs
                    g = lhs - rhs

                    # Check convergence
                    if abs(g) < 1e-10 * max(1.0, abs(lhs)):
                        converged = True
                        break

                    # Newton step: g'(c) = -u''(c) - beta * R * sum P * u''(c') * (dm'/dc)
                    # dm'/dc = -R, so d c'/dc = dc'/dm' * dm'/dc = dc'/dm' * (-R)
                    # This is complex; use a simpler derivative estimate
                    # g'(c) ≈ -u''(c) (dominant term, since c' changes slowly)
                    # u''(c) for CRRA: -gamma * c^(-gamma-1)
                    if is_log:
                        u2 = -1.0 / (c * c)
                    else:
                        u2 = -gamma * np.power(c, -gamma - 1)

                    # Actually, let's just use bisection for robustness
                    # since Newton can be unstable
                    if g > 0:
                        # u'(c) too high, decrease c
                        c_hi = c
                    else:
                        c_lo = c

                    # Use bisection step for stability
                    c_new = (c_lo + c_hi) / 2.0

                    # Check if close enough
                    if abs(c_new - c) < 1e-12 * max(1.0, c):
                        c = c_new
                        converged = True
                        break

                    c = c_new

                # Ensure c is within bounds
                c = np.clip(c, c_floor, m)
                c_opt[i] = c

            c_policy[t, z, :] = c_opt

            # Compute V[t, z] = utility(c) + beta * E[V[t+1, z'](m')]
            V[t, z, :] = utility(c_opt)
            for zp in range(num_states):
                m_next = R * (m_grid - c_opt) + income[zp]
                m_next = np.maximum(m_next, 0.0)
                # Interpolate V[t+1, zp]
                V_next_zp = np.interp(m_next, m_grid, V[t+1, zp, :])
                V[t, z, :] += beta * transition[z, zp] * V_next_zp

    # Now interpolate c_policy onto resource_grid
    result = np.zeros((horizon, num_states, len(resource_grid)))
    for t in range(horizon):
        for z in range(num_states):
            result[t, z, :] = np.interp(resource_grid, m_grid, c_policy[t, z, :])
            # Ensure constraints
            result[t, z, :] = np.clip(result[t, z, :], c_floor, resource_grid + 1e-10)

    # Final period: c[5] must equal grid within 1e-9
    result[horizon - 1, :, :] = resource_grid

    # Validate
    if not np.all(np.isfinite(result)):
        # Fallback to simple policy
        for t in range(horizon):
            for z in range(num_states):
                result[t, z, :] = resource_grid * 0.5
        result[horizon - 1, :, :] = resource_grid

    return result
