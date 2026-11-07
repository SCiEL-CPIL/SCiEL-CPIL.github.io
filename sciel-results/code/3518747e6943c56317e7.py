"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 135  # Safety margin below 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    m_out = resource_grid  # 101-point output grid
    n_out = len(m_out)
    n_z = len(income)

    # Dense internal grid for accurate continuation value interpolation
    m_dense = np.geomspace(0.001, 40, 4000)
    n_dense = len(m_dense)

    # Utility and derivatives
    if abs(gamma - 1.0) < 1e-10:
        def u(c):
            c = np.maximum(c, 1e-12)
            return np.log(c)
        def u_prime(c):
            c = np.maximum(c, 1e-12)
            return 1.0 / c
        def u_double_prime(c):
            c = np.maximum(c, 1e-12)
            return -1.0 / (c * c)
    else:
        def u(c):
            c = np.maximum(c, 1e-12)
            return c ** (1 - gamma) / (1 - gamma)
        def u_prime(c):
            c = np.maximum(c, 1e-12)
            return c ** (-gamma)
        def u_double_prime(c):
            c = np.maximum(c, 1e-12)
            return -gamma * c ** (-gamma - 1)

    # Continuation value V[z, m] for next period
    # V is shape (n_z, n_dense)
    V = np.zeros((n_z, n_dense))

    # Consumption policy on dense grid
    C_dense = np.zeros((horizon, n_z, n_dense))

    # Backward induction using Euler equation root-finding
    for t in range(horizon - 1, -1, -1):
        remaining = horizon - 1 - t  # periods left after current
        for z in range(n_z):
            m_vals = m_dense
            c_vals = np.zeros(n_dense)

            if remaining == 0:
                # Terminal: consume everything
                c_vals[:] = m_vals
            else:
                # For each m, find c* satisfying Euler equation:
                # u'(c) = beta * sum_z' P(z'|z) * u'(c')
                # where c' = R*(m-c) + income[z']
                #
                # We solve via Newton's method on:
                # f(c) = u'(c) - beta * E[u'(R*(m-c) + y')]
                #
                # For each m, do vectorized Newton over a batch.
                # Use a simple approach: for each m, Newton with a few iterations.
                # To speed up, vectorize over m.

                # Precompute: for a given c vector, compute expected next-period marginal utility
                # c' for each (m, z'): R*(m - c) + income[z']
                # We need: g(c) = beta * sum_z' P(z'|z) * u_prime(R*(m-c) + income[z'])

                # Newton: c_{k+1} = c_k - f(c_k)/f'(c_k)
                # f(c) = u'(c) - g(c)
                # f'(c) = u''(c) - g'(c)
                # g'(c) = beta * sum_z' P(z'|z) * u''(c') * (-R)
                #        = -beta * R * sum_z' P(z'|z) * u''(c')
                # So f'(c) = u''(c) + beta * R * sum_z' P(z'|z) * u''(c')

                # Initialize c: use a heuristic. For CRRA with gamma=1, c* ~ m * (1 - beta*R) / (1 - beta*R) ...
                # Simple init: c = m * (1 - beta) * something, or just m * 0.5
                # Better: for log utility, c* = m / (1 + beta*R) approximately when income is constant
                # Let's use c_init = m * (1 - beta) / (1 - beta * R) clipped to [1e-6, m]
                # Actually, a safe init: c = m * 0.5, then Newton
                c = np.clip(m_vals * 0.5, 1e-6, m_vals * 0.999)

                for newton_iter in range(20):
                    # Compute c' for each z'
                    # c' shape: (n_z, n_dense)
                    cp = R * (m_vals[None, :] - c[None, :]) + income[:, None]
                    cp = np.maximum(cp, 1e-12)

                    # g(c) = beta * sum_z' P(z'|z) * u'(cp)
                    up_cp = u_prime(cp)  # (n_z, n_dense)
                    g = beta * np.sum(transition[z, :, None] * up_cp, axis=0)  # (n_dense,)

                    # f(c) = u'(c) - g
                    up_c = u_prime(c)
                    f = up_c - g

                    # f'(c) = u''(c) - g'(c)
                    # g'(c) = beta * sum_z' P(z'|z) * u''(cp) * (-R)
                    upp_cp = u_double_prime(cp)  # (n_z, n_dense)
                    g_prime = beta * np.sum(transition[z, :, None] * upp_cp, axis=0) * (-R)
                    upp_c = u_double_prime(c)
                    f_prime = upp_c - g_prime

                    # Newton step
                    step = f / f_prime
                    c_new = c - step

                    # Clamp to valid range: 1e-6 <= c <= m
                    c_new = np.clip(c_new, 1e-6, m_vals * 0.9999)

                    # Check convergence
                    if np.max(np.abs(c_new - c)) < 1e-10:
                        c = c_new
                        break
                    c = c_new

                c_vals[:] = c

            C_dense[t, z, :] = c_vals

            # Check deadline
            if time.time() > deadline - 5:
                # Fill remaining with terminal-like policy
                for tt in range(t - 1, -1, -1):
                    for zz in range(n_z):
                        C_dense[tt, zz, :] = m_dense
                break

        # Compute V[z, m] = u(c*) + beta * E[V_next(z', c')]
        if remaining == 0:
            V[z, :] = u(c_vals)
        else:
            # c' for each z'
            cp = R * (m_dense[None, :] - c_vals[None, :]) + income[:, None]
            cp = np.maximum(cp, 1e-12)
            # Interpolate V_next at cp
            # V_next[z', m] is stored in V (which we're filling from the back)
            # Since we go backward, V for period t+1 is already computed
            V_next_interp = np.zeros((n_z, n_dense))
            for zp in range(n_z):
                V_next_interp[zp, :] = np.interp(cp[zp, :], m_dense, V[zp, :])
            # Wait - V is being filled. At time t, we need V for t+1.
            # Let me restructure: store V for each period.
            # Actually, let me use a 3D array V[t, z, m]
            # For now, let's just compute the value directly.
            # V[t, z, m] = u(c*) + beta * sum_z' P(z'|z) * V[t+1, z', c']
            # But we're computing in place. Let me use a separate array.
            pass

    # Actually, let me restructure to store V per period for clarity
    # Re-do with proper V storage
    V_all = np.zeros((horizon + 1, n_z, n_dense))  # V[horizon] = 0 (terminal)
    C_all = np.zeros((horizon, n_z, n_dense))

    # Terminal: V[horizon] = 0, but at t=horizon-1, c* = m (consume all)
    # Actually at t = horizon-1 (last period), you consume everything: c* = m
    # V[horizon-1, z, m] = u(m) + beta * E[0] = u(m)
    # Wait, the problem says c[5] (t=5, which is horizon-1 since horizon=6) equals the grid.
    # So at t=5 (last), c = m.

    for t in range(horizon - 1, -1, -1):
        for z in range(n_z):
            m_vals = m_dense
            if t == horizon - 1:
                # Last period: consume everything
                c_vals = m_vals.copy()
            else:
                # Newton on Euler equation
                c = np.clip(m_vals * 0.5, 1e-6, m_vals * 0.999)
                for newton_iter in range(30):
                    cp = R * (m_vals[None, :] - c[None, :]) + income[:, None]
                    cp = np.maximum(cp, 1e-12)

                    up_cp = u_prime(cp)
                    g = beta * np.sum(transition[z, :, None] * up_cp, axis=0)

                    up_c = u_prime(c)
                    f = up_c - g

                    upp_cp = u_double_prime(cp)
                    g_prime = beta * np.sum(transition[z, :, None] * upp_cp, axis=0) * (-R)
                    upp_c = u_double_prime(c)
                    f_prime = upp_c - g_prime

                    step = f / f_prime
                    c_new = c - step
                    c_new = np.clip(c_new, 1e-6, m_vals * 0.9999)

                    if np.max(np.abs(c_new - c)) < 1e-12:
                        c = c_new
                        break
                    c = c_new
                c_vals = c

            C_all[t, z, :] = c_vals

            # Compute V[t, z, m]
            if t == horizon - 1:
                V_all[t, z, :] = u(c_vals)
            else:
                cp = R * (m_dense[None, :] - c_vals[None, :]) + income[:, None]
                cp = np.maximum(cp, 1e-12)
                V_next_interp = np.zeros((n_z, n_dense))
                for zp in range(n_z):
                    V_next_interp[zp, :] = np.interp(cp[zp, :], m_dense, V_all[t + 1, zp, :])
                cont = beta * np.sum(transition[z, :, None] * V_next_interp, axis=0)
                V_all[t, z, :] = u(c_vals) + cont

            if time.time() > deadline - 5:
                break

        if time.time() > deadline - 5:
            break

    # Interpolate C_all from dense grid to output grid
    C_out = np.zeros((horizon, n_z, n_out))
    for t in range(horizon):
        for z in range(n_z):
            C_out[t, z, :] = np.interp(m_out, m_dense, C_all[t, z, :])

    # Enforce constraints: 0 < c <= m + 1e-10, and c[5] == grid within 1e-9
    for t in range(horizon):
        for z in range(n_z):
            C_out[t, z, :] = np.maximum(C_out[t, z, :], 1e-12)
            C_out[t, z, :] = np.minimum(C_out[t, z, :], m_out + 1e-10)
    # Terminal: c[5] must equal grid
    C_out[horizon - 1, :, :] = m_out[None, :]

    return C_out
# EVOLVE-BLOCK-END
