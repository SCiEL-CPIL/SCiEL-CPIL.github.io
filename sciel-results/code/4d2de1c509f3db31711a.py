"""Consumption-saving policy: FOC root-finding with vectorized continuation values."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(c)
    return c ** (1.0 - gamma) / (1.0 - gamma)

def _utility_prime(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return 1.0 / c
    return c ** (-gamma)

def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin under 120s/150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model.get('R', 1.02)
    income = np.asarray(model['income'], dtype=float)
    Tmat = np.asarray(model['transition'], dtype=float)
    horizon = int(model['horizon'])
    m_grid = np.asarray(resource_grid, dtype=float)

    n_z = len(income)
    n_m = len(m_grid)
    C = np.empty((horizon, n_z, n_m), dtype=float)

    # V_next[z, m] and Vp_next[z, m] = dV/dm at period t+1
    # Initialize with terminal condition: at t=horizon-1, consume all, V = u(m), Vp = u'(m)
    # But we work backward from horizon-1 down to 0.
    # At the last period (t = horizon-1), c = m, so V_{horizon-1}[z, m] = u(m)
    # For t < horizon-1, we use V_{t+1} which is already computed.
    # So we need V_next to represent V at period t+1.
    # Initialize: before the loop, V_next should be the terminal value function.
    # Actually, let's structure: loop t from horizon-1 down to 0.
    # When t == horizon-1: C = m_grid (terminal), then compute V_next for this period = u(m), Vp = u'(m)
    # When t < horizon-1: use V_next (which is V_{t+1}) to solve FOC, then compute V_t, Vp_t.

    V_next = np.zeros((n_z, n_m), dtype=float)
    Vp_next = np.zeros((n_z, n_m), dtype=float)

    for t in range(horizon - 1, -1, -1):
        if t == horizon - 1:
            # Terminal: consume all
            for z in range(n_z):
                C[t, z, :] = m_grid
            # Compute V and Vp for this period
            for z in range(n_z):
                V_next[z, :] = _utility(m_grid, gamma)
                Vp_next[z, :] = _utility_prime(m_grid, gamma)
            continue

        # For each z, solve for c(m) using FOC
        for z in range(n_z):
            probs = Tmat[z]

            # Precompute: for each m in m_grid, we need to solve
            # u'(c) = beta * R * sum_{z'} probs[z'] * Vp_next[z'](R*(m-c) + income[z'])
            #
            # Vectorize over m: for a given c vector (same length as m_grid),
            # compute m_next[z'] = R*(m_grid - c) + income[z'] for each z'
            # Then interpolate Vp_next[z'] at those m_next values.

            # Use iterative bisection over c for all m simultaneously.
            # c_low = 1e-10, c_high = m_grid - 1e-12
            c_lo = np.full(n_m, 1e-10)
            c_hi = np.maximum(m_grid - 1e-12, 1e-10)

            # Ensure c_hi > c_lo
            valid = c_hi > c_lo
            if not np.any(valid):
                C[t, z, :] = m_grid
                continue

            # Bisection iterations
            for _iter in range(50):
                c_mid = 0.5 * (c_lo + c_hi)

                # Compute m_next for each z'
                # m_next shape: (n_z, n_m)
                m_next = R * (m_grid[np.newaxis, :] - c_mid[np.newaxis, :]) + income[:, np.newaxis]
                # Clip to grid range
                m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])

                # Interpolate Vp_next for each z' at m_next_clipped
                # Evp[z', :] = interp(m_next_clipped[z', :], m_grid, Vp_next[z', :])
                Evp = np.array([np.interp(m_next_clipped[zp, :], m_grid, Vp_next[zp, :]) for zp in range(n_z)])
                # Expected Vp: shape (n_m,)
                expected_vp = probs @ Evp  # (n_z,) @ (n_z, n_m) -> (n_m,)

                # f(c) = u'(c) - beta * R * expected_vp
                f = _utility_prime(c_mid, gamma) - beta * R * expected_vp

                # Update bounds
                # f decreasing in c: if f > 0, c too low -> c_lo = c_mid
                # if f < 0, c too high -> c_hi = c_mid
                c_lo = np.where(f > 0, c_mid, c_lo)
                c_hi = np.where(f > 0, c_hi, c_mid)

                # Check convergence
                if np.all(np.abs(c_hi - c_lo) < 1e-12):
                    break

            C[t, z, :] = 0.5 * (c_lo + c_hi)

            # Deadline guard
            if time.time() > deadline:
                # Fill remaining
                for zz in range(z + 1, n_z):
                    C[t, zz, :] = m_grid
                # Compute V for completed z's and return
                _compute_V(C, t, z, n_z, n_m, m_grid, income, Tmat, beta, gamma, R, V_next, Vp_next)
                return C

        # Compute V_curr and Vp_curr for this period
        V_curr = np.zeros((n_z, n_m), dtype=float)
        Vp_curr = np.zeros((n_z, n_m), dtype=float)

        for z in range(n_z):
            probs = Tmat[z]
            c = C[t, z, :]
            m_next = R * (m_grid[np.newaxis, :] - c[np.newaxis, :]) + income[:, np.newaxis]
            m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])

            # Interpolate V and Vp
            Ev = np.array([np.interp(m_next_clipped[zp, :], m_grid, V_next[zp, :]) for zp in range(n_z)])
            Evp = np.array([np.interp(m_next_clipped[zp, :], m_grid, Vp_next[zp, :]) for zp in range(n_z)])

            expected_V = probs @ Ev
            expected_Vp = probs @ Evp

            V_curr[z, :] = _utility(c, gamma) + beta * expected_V
            Vp_curr[z, :] = beta * R * expected_Vp

        V_next = V_curr
        Vp_next = Vp_curr

        # Deadline guard
        if time.time() > deadline:
            return C

    # Final validity checks
    C = np.clip(C, 1e-10, None)
    # Ensure c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            C[t, z, :] = np.minimum(C[t, z, :], m_grid + 1e-10)
    # Terminal condition: c[5] = m_grid
    C[horizon - 1, :, :] = m_grid

    return C

def _compute_V(C, t, n_z_done, n_z, n_m, m_grid, income, Tmat, beta, gamma, R, V_next, Vp_next):
    """Partial V computation for deadline fallback."""
    # This is a simplified fallback; just return what we have
    pass

# EVOLVE-BLOCK-END
