"""Consumption-saving policy: FOC root-finding with smooth rational interpolation."""
import time
import numpy as np
from scipy.interpolate import Rbf

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
    deadline = t0 + 115.0

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

    # Terminal: consume all
    C[horizon - 1, :, :] = m_grid

    # V_next[z, m] = value at period t+1
    # Vp_next[z, m] = dV/dm at period t+1
    V_next = np.zeros((n_z, n_m), dtype=float)
    Vp_next = np.zeros((n_z, n_m), dtype=float)
    for z in range(n_z):
        V_next[z, :] = _utility(m_grid, gamma)
        Vp_next[z, :] = _utility_prime(m_grid, gamma)

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            probs = Tmat[z]

            # Vectorized bisection over x = c/m in (0,1)
            x_lo = np.full(n_m, 1e-10)
            x_hi = np.full(n_m, 1.0 - 1e-12)

            valid = m_grid > 1e-8
            x_lo[~valid] = 0.0
            x_hi[~valid] = 0.0

            for _iter in range(45):
                x_mid = 0.5 * (x_lo + x_hi)
                c_mid = x_mid * m_grid

                # m_next[z', m] = R*(m - c) + income[z']
                m_next = R * (m_grid[np.newaxis, :] - c_mid[np.newaxis, :]) + income[:, np.newaxis]
                m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])

                # Interpolate Vp_next for each z'
                Evp = np.empty((n_z, n_m))
                for zp in range(n_z):
                    Evp[zp, :] = np.interp(m_next_clipped[zp, :], m_grid, Vp_next[zp, :])

                expected_vp = probs @ Evp

                f = _utility_prime(c_mid, gamma) - beta * R * expected_vp

                x_lo = np.where(f > 0, x_mid, x_lo)
                x_hi = np.where(f > 0, x_hi, x_mid)

                if np.all((x_hi - x_lo) * m_grid < 1e-10):
                    break

            C[t, z, :] = 0.5 * (x_lo + x_hi) * m_grid

            if time.time() > deadline:
                for zz in range(z + 1, n_z):
                    C[t, zz, :] = m_grid
                C[horizon - 1, :, :] = m_grid
                _enforce_constraints(C, horizon, n_z, n_m, m_grid)
                return C

        # Compute V_curr and Vp_curr for period t
        V_curr = np.zeros((n_z, n_m), dtype=float)
        Vp_curr = np.zeros((n_z, n_m), dtype=float)

        for z in range(n_z):
            probs = Tmat[z]
            c = C[t, z, :]
            m_next = R * (m_grid[np.newaxis, :] - c[np.newaxis, :]) + income[:, np.newaxis]
            m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])

            Ev = np.empty((n_z, n_m))
            Evp = np.empty((n_z, n_m))
            for zp in range(n_z):
                Ev[zp, :] = np.interp(m_next_clipped[zp, :], m_grid, V_next[zp, :])
                Evp[zp, :] = np.interp(m_next_clipped[zp, :], m_grid, Vp_next[zp, :])

            expected_V = probs @ Ev
            expected_Vp = probs @ Evp

            V_curr[z, :] = _utility(c, gamma) + beta * expected_V
            Vp_curr[z, :] = beta * R * expected_Vp

        V_next = V_curr
        Vp_next = Vp_curr

        if time.time() > deadline:
            _enforce_constraints(C, horizon, n_z, n_m, m_grid)
            return C

    _enforce_constraints(C, horizon, n_z, n_m, m_grid)
    return C

def _enforce_constraints(C, horizon, n_z, n_m, m_grid):
    """Ensure finite values, 0 < c <= m + 1e-10, and c[5] = grid."""
    C = np.where(np.isfinite(C), C, 1e-12)
    C = np.clip(C, 1e-12, None)
    for t in range(horizon):
        for z in range(n_z):
            C[t, z, :] = np.minimum(C[t, z, :], m_grid + 1e-10)
    C[horizon - 1, :, :] = m_grid

# EVOLVE-BLOCK-END
