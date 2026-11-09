"""Combined program: log-linear value approximation with homothetic FOC solver."""
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

    # For the terminal period, V(m) = u(m), V'(m) = u'(m)
    # We'll store V and Vp on the grid for interpolation, but also try
    # to use a log-linear approximation for smoother derivatives.
    # 
    # Strategy: 
    # 1. Compute exact V and Vp on the grid via backward induction.
    # 2. At each step, when computing the FOC, interpolate Vp from the grid.
    #    But to improve accuracy on the finer scoring grid, we can fit a
    #    log-linear approximation V(m) ≈ A + B*log(m) to the grid values,
    #    and use its derivative B/m for the FOC. This gives a smooth,
    #    analytically exact derivative that generalizes well.
    #
    # Actually, the key insight from the directive is to use the log-linear
    # approximation for the *continuation value derivative* in the FOC,
    # which avoids interpolation errors on the coarse grid.
    #
    # Let's do a hybrid: compute V on the grid, fit log-linear, and use
    # the fitted derivative for the FOC. This should be more accurate
    # when the policy is interpolated onto the 251-point grid.

    V_next = np.zeros((n_z, n_m), dtype=float)
    Vp_next = np.zeros((n_z, n_m), dtype=float)
    for z in range(n_z):
        V_next[z, :] = _utility(m_grid, gamma)
        Vp_next[z, :] = _utility_prime(m_grid, gamma)

    # Log-linear coefficients: V(z, m) ≈ A[z] + B[z] * log(m)
    # Fit these to the grid values.
    log_m = np.log(m_grid)

    def fit_log_linear(V_z):
        """Fit V ≈ A + B*log(m) to V_z on m_grid."""
        # Least squares: [1, log_m] @ [A, B] = V_z
        X = np.column_stack([np.ones(n_m), log_m])
        coeffs, _, _, _ = np.linalg.lstsq(X, V_z, rcond=None)
        return coeffs[0], coeffs[1]

    # Initialize log-linear fits
    A_next = np.zeros(n_z)
    B_next = np.zeros(n_z)
    for z in range(n_z):
        A_next[z], B_next[z] = fit_log_linear(V_next[z, :])

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            probs = Tmat[z]

            # FOC: u'(c) = beta * R * E[V'(m')]
            # where m' = R*(m - c) + income[z']
            # Using log-linear approximation: V'(m') ≈ B[z'] / m'
            # So E[V'(m')] ≈ sum_z' probs[z'] * B_next[z'] / (R*(m-c) + income[z'])
            #
            # This is a smooth function of c, and we can solve for c via bisection
            # on x = c/m.

            x_lo = np.full(n_m, 1e-10)
            x_hi = np.full(n_m, 1.0 - 1e-12)

            valid = m_grid > 1e-8
            x_lo[~valid] = 0.0
            x_hi[~valid] = 0.0

            for _iter in range(50):
                x_mid = 0.5 * (x_lo + x_hi)
                c_mid = x_mid * m_grid

                # m_next[z', m] = R*(m - c) + income[z']
                m_next = R * (m_grid[np.newaxis, :] - c_mid[np.newaxis, :]) + income[:, np.newaxis]
                # Ensure m_next is positive for log
                m_next_safe = np.maximum(m_next, 1e-12)

                # E[V'(m')] using log-linear approx: sum_z' probs[z'] * B_next[z'] / m_next[z']
                # B_next[z'] / m_next[z', :] has shape (n_z, n_m)
                evp = np.sum(probs[:, np.newaxis] * (B_next[:, np.newaxis] / m_next_safe), axis=0)

                f = _utility_prime(c_mid, gamma) - beta * R * evp

                x_lo = np.where(f > 0, x_mid, x_lo)
                x_hi = np.where(f > 0, x_hi, x_mid)

                if np.all((x_hi - x_lo) * m_grid < 1e-10):
                    break

            C[t, z, :] = 0.5 * (x_lo + x_hi) * m_grid

            # Deadline guard
            if time.time() > deadline:
                for zz in range(z + 1, n_z):
                    C[t, zz, :] = m_grid
                C[horizon - 1, :, :] = m_grid
                _enforce_constraints(C, m_grid, horizon, n_z)
                return C

        # Compute V_curr and update log-linear fits
        V_curr = np.zeros((n_z, n_m), dtype=float)
        Vp_curr = np.zeros((n_z, n_m), dtype=float)

        for z in range(n_z):
            probs = Tmat[z]
            c = C[t, z, :]
            m_next = R * (m_grid[np.newaxis, :] - c[np.newaxis, :]) + income[:, np.newaxis]
            m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])

            # Use grid interpolation for V (more accurate than log-linear for V itself)
            Ev = np.empty((n_z, n_m))
            for zp in range(n_z):
                Ev[zp, :] = np.interp(m_next_clipped[zp, :], m_grid, V_next[zp, :])

            expected_V = probs @ Ev
            V_curr[z, :] = _utility(c, gamma) + beta * expected_V

            # Vp via envelope theorem: Vp(z, m) = beta * R * E[Vp(z', m')]
            # Use log-linear for Vp: B_next[z'] / m'
            m_next_safe = np.maximum(m_next, 1e-12)
            evp = np.sum(probs[:, np.newaxis] * (B_next[:, np.newaxis] / m_next_safe), axis=0)
            Vp_curr[z, :] = beta * R * evp

        # Update log-linear fits
        A_next = np.zeros(n_z)
        B_next = np.zeros(n_z)
        for z in range(n_z):
            A_next[z], B_next[z] = fit_log_linear(V_curr[z, :])

        V_next = V_curr
        Vp_next = Vp_curr

        if time.time() > deadline:
            _enforce_constraints(C, m_grid, horizon, n_z)
            return C

    # Enforce constraints
    _enforce_constraints(C, m_grid, horizon, n_z)
    return C

def _enforce_constraints(C, m_grid, horizon, n_z):
    C = np.clip(C, 1e-12, None)
    for t in range(horizon):
        for z in range(n_z):
            C[t, z, :] = np.minimum(C[t, z, :], m_grid + 1e-10)
    C[horizon - 1, :, :] = m_grid
    # Ensure finite
    C = np.where(np.isfinite(C), C, m_grid[np.newaxis, np.newaxis, :])
    return C

# EVOLVE-BLOCK-END
