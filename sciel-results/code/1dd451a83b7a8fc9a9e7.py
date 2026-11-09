"""Rewritten consumption policy: vectorized backward induction + Newton polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return np.log(c)
    return np.power(c, 1.0 - gamma) / (1.0 - gamma)


def _u_prime(c, gamma):
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return 1.0 / c
    return np.power(c, -gamma)


def _u_prime2(c, gamma):
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return -1.0 / (c * c)
    return -gamma * np.power(c, -gamma - 1.0)


def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    DEADLINE = 150.0
    SAFETY = 5.0  # seconds before deadline to stop

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    P = np.asarray(model['transition'], dtype=float)
    H = int(model['horizon'])
    n_states = len(income)
    m_grid = np.asarray(resource_grid, dtype=float)
    n_m = len(m_grid)

    # consumption[t, z, i]
    c = np.zeros((H, n_states, n_m), dtype=float)
    V = np.zeros((H + 1, n_states, n_m), dtype=float)

    # Terminal: consume all wealth
    c[H - 1] = m_grid[None, None, :]
    V[H] = 0.0  # V[H] is continuation after last period = 0

    # Backward induction from t = H-2 down to 0
    # We use a fine grid of consumption fractions
    # The seed uses 48 points; we can use more since it's fast
    # Use a denser grid: 128 fractions
    n_frac = 128
    fracs = np.linspace(0.001, 1.0, n_frac)  # fraction of m to consume

    for t in range(H - 2, -1, -1):
        if time.time() > DEADLINE - SAFETY:
            break
        # V_next[t+1, z', m'] for each state
        # For each state z and grid m, try each consumption fraction
        # c = m * frac, m' = R*(m - c) + income[z']
        # objective = u(c) + beta * sum_{z'} P[z, z'] * V_next[z', m']
        # V_next is on m_grid; we need to interpolate m' onto m_grid

        # Vectorize over (z, m, frac)
        # m shape (n_m,), frac shape (n_frac,)
        # c = m[:, None] * fracs[None, :]  -> (n_m, n_frac)
        # m_minus_c = m[:, None] - c -> (n_m, n_frac)
        # m_next = R * m_minus_c + income[z'] -> for each z'
        # We need V_next[z', m_next] via interpolation

        # Build: for each z (state), compute objective over (m, frac)
        V_next = V[t + 1]  # (n_states, n_m)

        best_obj = np.zeros((n_states, n_m))
        best_c = np.zeros((n_states, n_m))

        # Precompute u(c) for all m, frac
        # c_mat = m_grid[:, None] * fracs[None, :]  (n_m, n_frac)
        c_mat = m_grid[:, None] * fracs[None, :]  # (n_m, n_frac)
        u_c = _utility(c_mat, gamma)  # (n_m, n_frac)

        # m_next for each z': R*(m - c) + income[z']
        # m_minus_c = m_grid[:, None] - c_mat  (n_m, n_frac)
        m_minus_c = m_grid[:, None] - c_mat  # (n_m, n_frac)

        for z in range(n_states):
            # Transition from state z to z'
            # For each z', m_next = R * m_minus_c + income[z']
            # V_next[z', m_next] interpolated
            # obj[m, frac] = u_c[m, frac] + beta * sum_{z'} P[z, z'] * V_next[z', m_next_z']

            # Weighted sum over z'
            # For each z', compute V_next[z', R*m_minus_c + income[z']]
            # This is (n_m, n_frac) for each z'
            weighted_V = np.zeros((n_m, n_frac))
            for zp in range(n_states):
                m_next = R * m_minus_c + income[zp]  # (n_m, n_frac)
                # Clip m_next to grid range for interpolation
                m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])
                V_interp = np.interp(m_next_clipped, m_grid, V_next[zp])  # (n_m, n_frac)
                weighted_V += P[z, zp] * V_interp

            obj = u_c + beta * weighted_V  # (n_m, n_frac)

            # Find best frac for each m
            best_idx = np.argmax(obj, axis=1)  # (n_m,)
            best_obj[z] = obj[np.arange(n_m), best_idx]
            best_c[z] = c_mat[np.arange(n_m), best_idx]

        V[t] = best_obj
        c[t] = best_c

    # Newton polish pass: refine c using first-order condition
    # FOC: u'(c) = beta * sum_z' P[z,z'] * R * V_{m'}(z', m')
    # where m' = R*(m-c) + income[z']
    # V_m' is the derivative of V with respect to m

    if time.time() < DEADLINE - SAFETY:
        # Compute V_m (derivative of value function w.r.t. m) via finite differences
        # or analytically: V_m = beta * sum P[z,z'] * R * V_m_next[z', m']
        # At terminal: V_m[H-1] = u'(m) / 1 (since c=m, dc/dm=1)
        # Actually V[H-1, z, m] = u(m), so V_m[H-1] = u'(m)

        V_m = np.zeros((H, n_states, n_m))
        V_m[H - 1] = _u_prime(m_grid, gamma)[None, None, :]

        for t in range(H - 2, -1, -1):
            if time.time() > DEADLINE - SAFETY:
                break
            # V_m[t, z, m] = beta * sum_z' P[z,z'] * R * V_m[t+1, z', m']
            # where m' = R*(m - c[t,z,m]) + income[z']
            for z in range(n_states):
                # c[t, z, :] shape (n_m,)
                ct = c[t, z, :]  # (n_m,)
                m_minus_c = m_grid - ct  # (n_m,)
                for zp in range(n_states):
                    m_next = R * m_minus_c + income[zp]  # (n_m,)
                    m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])
                    Vm_interp = np.interp(m_next_clipped, m_grid, V_m[t + 1, zp])
                    if z == 0:
                        V_m[t, z, :] = beta * P[z, zp] * R * Vm_interp
                    else:
                        V_m[t, z, :] += beta * P[z, zp] * R * Vm_interp

        # Newton iterations
        n_newton = 8
        for _ in range(n_newton):
            if time.time() > DEADLINE - SAFETY:
                break
            for t in range(H - 1):  # t = 0 to H-2 (H-1 is terminal, c=m)
                if time.time() > DEADLINE - SAFETY:
                    break
                for z in range(n_states):
                    ct = c[t, z, :].copy()  # (n_m,)
                    m_grid_local = m_grid  # (n_m,)

                    # Compute RHS: beta * sum_z' P[z,z'] * R * V_m[t+1, z', m']
                    # where m' = R*(m - c) + income[z']
                    rhs = np.zeros(n_m)
                    for zp in range(n_states):
                        m_next = R * (m_grid_local - ct) + income[zp]
                        m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])
                        Vm_interp = np.interp(m_next_clipped, m_grid, V_m[t + 1, zp])
                        rhs += P[z, zp] * Vm_interp
                    rhs *= beta * R

                    # Newton: solve u'(c) = rhs
                    # f(c) = u'(c) - rhs = 0
                    # f'(c) = u''(c)
                    up = _u_prime(ct, gamma)
                    up2 = _u_prime2(ct, gamma)
                    # Newton step: c_new = c - f/f' = c - (up - rhs)/up2
                    # up2 is negative, so this should converge
                    delta_c = (up - rhs) / up2
                    ct = ct - delta_c

                    # Enforce constraints: 0 < c <= m
                    ct = np.maximum(ct, 1e-12)
                    ct = np.minimum(ct, m_grid_local + 1e-10)

                    c[t, z, :] = ct

        # Recompute V_m after Newton for consistency (optional, skip if time-tight)
        # Recompute V after Newton
        if time.time() < DEADLINE - SAFETY:
            V[H] = 0.0
            c[H - 1] = m_grid[None, None, :]
            V[H - 1] = _utility(m_grid, gamma)[None, None, :]
            V_m[H - 1] = _u_prime(m_grid, gamma)[None, None, :]

            for t in range(H - 2, -1, -1):
                if time.time() > DEADLINE - SAFETY:
                    break
                for z in range(n_states):
                    ct = c[t, z, :]
                    m_minus_c = m_grid - ct
                    obj = _utility(ct, gamma)
                    for zp in range(n_states):
                        m_next = R * m_minus_c + income[zp]
                        m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])
                        V_interp = np.interp(m_next_clipped, m_grid, V[t + 1, zp])
                        obj += beta * P[z, zp] * V_interp
                    V[t, z, :] = obj

                    # V_m
                    rhs = np.zeros(n_m)
                    for zp in range(n_states):
                        m_next = R * m_minus_c + income[zp]
                        m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])
                        Vm_interp = np.interp(m_next_clipped, m_grid, V_m[t + 1, zp])
                        rhs += P[z, zp] * Vm_interp
                    V_m[t, z, :] = beta * R * rhs

    # Final validation: ensure 0 < c <= m + 1e-10 and c[5] = grid
    c = np.maximum(c, 1e-12)
    c = np.minimum(c, m_grid[None, None, :] + 1e-10)
    c[H - 1] = m_grid[None, None, :]

    # Ensure finite
    c[~np.isfinite(c)] = m_grid[None, None, :][~np.isfinite(c)]

    return c
# EVOLVE-BLOCK-END
