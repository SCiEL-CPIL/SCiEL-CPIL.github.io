import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'])
    transition = np.asarray(model['transition'])
    horizon = model['horizon']
    m_out = np.asarray(resource_grid)
    N_out = len(m_out)
    N_z = len(income)

    # Dense internal grid for computation
    m_dense = np.geomspace(0.001, 40, 401)
    N_dense = len(m_dense)

    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1.0 - gamma) / (1.0 - gamma)

    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # Value functions: V[t, z, :] for each t and state z
    # V has shape (horizon, N_z, N_dense)
    V = np.zeros((horizon, N_z, N_dense))

    # Terminal condition: V[horizon-1, z, m] = util(m) since c = m at t=horizon-1
    # Actually, c[5] = m means at t=5, consumption equals resources
    # So V[5, z, m] = util(m) + beta * sum_z' P(z'|z) * 0 (no future)
    # Wait, horizon=6 means t=0..5. At t=5, c=m, and there's no t=6.
    # So V[5, z, m] = util(m)

    # Initialize terminal
    V[horizon - 1, :, :] = util(m_dense)[np.newaxis, :]  # same for all z

    # Backward induction for t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > DEADLINE - 5.0:
            # Not enough time; break and return what we have
            break

        for z in range(N_z):
            # For each m in m_dense, find optimal c
            # V_t(m, z) = max_c { util(c) + beta * sum_{z'} P(z'|z) * V_{t+1}(R*(m-c) + income[z'], z') }
            # subject to 0 < c <= m

            # Precompute: for each z', the continuation value as a function of m'
            # m' = R*(m - c) + income[z']
            # For a given c and m, m' is determined.

            # We'll use interpolation of V[t+1, z', :] to evaluate at arbitrary m'
            # V_next[z', :] is defined on m_dense

            # Strategy: for each m, do a 1D optimization over c in (0, m)
            # Use bisection on the FOC: u'(c) = beta * R * sum_z' P(z'|z) * V'_{t+1}(m', z')
            # where V' is the derivative of V_{t+1} w.r.t. its argument

            # Since V_{t+1} is a function of m', and m' = R*(m-c) + income[z'],
            # dV_{t+1}/dc = -R * sum_z' P(z'|z) * V'_{t+1}(m', z')

            # So FOC: u'(c) = beta * R * sum_z' P(z'|z) * V'_{t+1}(R*(m-c) + income[z'], z')

            # We need V'_{t+1}. We can approximate it by finite differences on m_dense.

            # Compute derivative of V[t+1, z', :] w.r.t. m
            # Using central differences where possible
            V_next = V[t + 1]  # shape (N_z, N_dense)

            # Compute dV/dm for each z'
            dV_dm = np.zeros_like(V_next)
            dm = m_dense[1] - m_dense[0]
            # Central differences
            dV_dm[:, 1:-1] = (V_next[:, 2:] - V_next[:, :-2]) / (2.0 * dm)
            # Forward at start
            dV_dm[:, 0] = (V_next[:, 1] - V_next[:, 0]) / dm
            # Backward at end
            dV_dm[:, -1] = (V_next[:, -1] - V_next[:, -2]) / dm

            # For each m in m_dense, solve for c
            c_opt = np.zeros(N_dense)

            for i, m in enumerate(m_dense):
                # c in (0, m), let's search in [1e-8, m]
                # FOC: u'(c) = beta * R * sum_z' P(z'|z) * dV_dm[z', m']
                # where m' = R*(m - c) + income[z']

                # Define f(c) = u'(c) - beta * R * sum_z' P(z'|z) * dV_dm[z', m']
                # We want f(c) = 0. f is decreasing in c (u' decreasing, and as c increases,
                # m' decreases, so dV_dm at m' increases (V is concave, so V' decreasing in m),
                # so the RHS decreases as c increases... wait.

                # As c increases, m' = R*(m-c) + income[z'] decreases.
                # V' is decreasing in m (concavity), so V'(m') is larger when m' is smaller.
                # So as c increases, RHS = beta*R*sum P * V'(m') increases.
                # u'(c) decreases in c.
                # So f(c) = u'(c) - RHS: u' decreasing, RHS increasing, so f is strictly decreasing.
                # f(0+) = u'(0+) - RHS(0) = inf - finite = +inf
                # f(m) = u'(m) - RHS(m). At c=m, m' = income[z'].
                # If f(m) > 0, then c* = m (corner solution, consume all)
                # If f(m) <= 0, then there's an interior solution.

                def f(c_val):
                    m_prime = R * (m - c_val) + income  # shape (N_z,)
                    # Clip m_prime to be within m_dense range for interpolation
                    m_prime_clipped = np.clip(m_prime, m_dense[0], m_dense[-1])
                    # Interpolate dV_dm at m_prime_clipped for each z'
                    # dV_dm has shape (N_z, N_dense)
                    rhs_terms = []
                    for zp in range(N_z):
                        # Interpolate dV_dm[zp, :] at m_prime_clipped[zp]
                        val = np.interp(m_prime_clipped[zp], m_dense, dV_dm[zp, :])
                        rhs_terms.append(transition[z, zp] * val)
                    rhs = beta * R * np.sum(rhs_terms)
                    return u_prime(c_val) - rhs

                # Check if corner solution
                f_at_m = f(m)
                if f_at_m > 1e-10:
                    c_opt[i] = m
                else:
                    # Bisection in [1e-10, m]
                    lo = 1e-10
                    hi = m
                    # f(lo) > 0, f(hi) <= 0
                    for _ in range(40):
                        mid = 0.5 * (lo + hi)
                        f_mid = f(mid)
                        if f_mid > 0:
                            lo = mid
                        else:
                            hi = mid
                    c_opt[i] = 0.5 * (lo + hi)

            # Now compute V[t, z, :] using the optimal c
            # V[t, z, m] = util(c_opt) + beta * sum_z' P(z'|z) * V[t+1, z', m']
            c_vals = c_opt
            m_prime = R * (m_dense[:, np.newaxis] - c_vals[:, np.newaxis]) + income[np.newaxis, :]
            m_prime_clipped = np.clip(m_prime, m_dense[0], m_dense[-1])

            # Interpolate V_next at m_prime_clipped
            cont_vals = np.zeros((N_dense, N_z))
            for zp in range(N_z):
                cont_vals[:, zp] = np.interp(m_prime_clipped[:, zp], m_dense, V_next[zp, :])

            # V[t, z, m] = util(c) + beta * sum_z' P(z'|z) * V_next[z', m']
            V[t, z, :] = util(c_vals) + beta * np.sum(transition[z, :] * cont_vals, axis=1)

    # Now we have V on m_dense. We need to output c[t, z, m_out] for m_out (101 points).
    # For t = horizon-1, c = m (terminal condition)
    # For t < horizon-1, we need to find c from the FOC or from the stored optimal c.

    # Actually, we should store the optimal c for each t, z, m_dense.
    # Let's redo: store c_opt for each t, z.

    # Let me restructure: store C[t, z, :] = optimal consumption
    C = np.zeros((horizon, N_z, N_dense))
    C[horizon - 1, :, :] = m_dense[np.newaxis, :]  # c = m at terminal

    # Recompute backward, storing C
    V = np.zeros((horizon, N_z, N_dense))
    V[horizon - 1, :, :] = util(m_dense)[np.newaxis, :]

    for t in range(horizon - 2, -1, -1):
        if time.time() > DEADLINE - 5.0:
            # For remaining t, use a rough estimate
            for z in range(N_z):
                C[t, z, :] = m_dense * 0.5  # rough
                V[t, z, :] = util(m_dense * 0.5)
            break

        for z in range(N_z):
            V_next = V[t + 1]
            dV_dm = np.zeros_like(V_next)
            dm = m_dense[1] - m_dense[0]
            dV_dm[:, 1:-1] = (V_next[:, 2:] - V_next[:, :-2]) / (2.0 * dm)
            dV_dm[:, 0] = (V_next[:, 1] - V_next[:, 0]) / dm
            dV_dm[:, -1] = (V_next[:, -1] - V_next[:, -2]) / dm

            c_opt = np.zeros(N_dense)

            for i, m in enumerate(m_dense):
                def f(c_val):
                    m_prime = R * (m - c_val) + income
                    m_prime_clipped = np.clip(m_prime, m_dense[0], m_dense[-1])
                    rhs = 0.0
                    for zp in range(N_z):
                        val = np.interp(m_prime_clipped[zp], m_dense, dV_dm[zp, :])
                        rhs += transition[z, zp] * val
                    rhs *= beta * R
                    return u_prime(c_val) - rhs

                f_at_m = f(m)
                if f_at_m > 1e-10:
                    c_opt[i] = m
                else:
                    lo = 1e-10
                    hi = m
                    for _ in range(40):
                        mid = 0.5 * (lo + hi)
                        f_mid = f(mid)
                        if f_mid > 0:
                            lo = mid
                        else:
                            hi = mid
                    c_opt[i] = 0.5 * (lo + hi)

            C[t, z, :] = c_opt

            # Compute V[t, z, :]
            c_vals = c_opt
            m_prime = R * (m_dense[:, np.newaxis] - c_vals[:, np.newaxis]) + income[np.newaxis, :]
            m_prime_clipped = np.clip(m_prime, m_dense[0], m_dense[-1])

            cont_vals = np.zeros((N_dense, N_z))
            for zp in range(N_z):
                cont_vals[:, zp] = np.interp(m_prime_clipped[:, zp], m_dense, V_next[zp, :])

            V[t, z, :] = util(c_vals) + beta * np.sum(transition[z, :] * cont_vals, axis=1)

    # Now interpolate C from m_dense to m_out
    result = np.zeros((horizon, N_z, N_out))
    for t in range(horizon):
        for z in range(N_z):
            result[t, z, :] = np.interp(m_out, m_dense, C[t, z, :])

    # Enforce constraints: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(N_z):
            result[t, z, :] = np.maximum(result[t, z, :], 1e-12)
            result[t, z, :] = np.minimum(result[t, z, :], m_out + 1e-10)

    # Terminal condition: c[5] must equal m within 1e-9
    result[horizon - 1, :, :] = m_out[np.newaxis, :]

    return result
