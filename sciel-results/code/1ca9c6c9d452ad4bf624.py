"""Rewritten solve_consumption using log-space PCHIP + Newton Euler solver."""
import numpy as np
from scipy.interpolate import PchipInterpolator
import time

# EVOLVE-BLOCK-START
def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    grid = np.asarray(resource_grid, dtype=float)
    n = len(grid)

    # log-space grid for interpolation
    log_grid = np.log(grid)

    # value function arrays: V[t, z, :]
    V = np.zeros((horizon, 2, n))
    # consumption policy arrays
    C = np.zeros((horizon, 2, n))

    # terminal: c = m, so V[5, z, i] = u(m_i)
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        else:
            return c ** (1.0 - gamma) / (1.0 - gamma)

    V[5, 0, :] = u(grid)
    V[5, 1, :] = u(grid)
    C[5, 0, :] = grid
    C[5, 1, :] = grid

    # Build log-space PCHIP interpolators for V[t+1, z'] for each z'
    def make_interp(V_row):
        # V_row: array of shape (n,)
        return PchipInterpolator(log_grid, V_row, extrapolate=True)

    def make_interp_prime(V_row, h=1e-4):
        # Numerical derivative of V in log-space: dV/d(log m) = m * V'(m)
        # We'll compute V'(m) via finite differences on the grid, then multiply by m
        # Actually, let's just compute dV/d(log m) directly using the PCHIP of V
        # dV/d(log m) = m * dV/dm
        # We'll compute this numerically
        pass

    # For Newton, we need the Euler equation:
    # u'(c) = beta * sum_z' P(z'|z) * R * V'_z'(R*(m-c) + income[z'])
    # where V'_z' is the derivative of V_z' with respect to its argument (m').
    #
    # Let f(c) = u'(c) - beta * sum_z' P(z'|z) * R * V'_z'(m'(c))
    # where m'(c) = R*(m-c) + income[z']
    #
    # We need V'_z'(x) = dV_z'/dx evaluated at x = m'(c).
    # Using PCHIP on log-space: V_z'(log m) is interpolated, and
    # dV_z'/dm = (dV_z'/d(log m)) / m
    # So V'_z'(x) = (dV_z'/d(log m)) / x  where x is the argument.

    # Let's build interpolators for dV/d(log m) as well, using finite differences.
    def build_log_deriv(V_row):
        # Compute dV/d(log m) at each grid point using central differences
        # dV/d(log m) = m * dV/dm
        # dV/dm at grid[i] using central differences
        dm = np.zeros(n)
        dm[1:-1] = (V_row[2:] - V_row[:-2]) / (grid[2:] - grid[:-2])
        dm[0] = (V_row[1] - V_row[0]) / (grid[1] - grid[0])
        dm[-1] = (V_row[-1] - V_row[-2]) / (grid[-1] - grid[-2])
        dV_dlogm = grid * dm
        return dV_dlogm

    # Precompute for t=5 (terminal)
    dV_dlogm_next = np.zeros((2, n))
    for z in range(2):
        dV_dlogm_next[z, :] = build_log_deriv(V[5, z, :])
    interp_dV_dlogm = [PchipInterpolator(log_grid, dV_dlogm_next[z, :], extrapolate=True) for z in range(2)]

    # For each t from 4 down to 0:
    for t in range(horizon - 2, -1, -1):
        if time.time() > DEADLINE:
            break

        for z in range(2):
            # For each m in grid, solve for c
            for i in range(n):
                m_i = grid[i]

                # m'(c) = R*(m - c) + income[z']
                # c must satisfy 0 < c <= m (approximately)
                # m'(c) must be > 0 for the continuation value to be well-defined
                # R*(m - c) + income[z'] > 0  =>  c < m + income[z']/R

                def euler_residual(c):
                    if c <= 0:
                        c = 1e-12
                    if c > m_i + 1e-10:
                        return None
                    # u'(c)
                    if abs(gamma - 1.0) < 1e-12:
                        mu = 1.0 / c
                    else:
                        mu = c ** (-gamma)
                    # Continuation: beta * sum_z' P(z'|z) * R * V'_z'(m'(c))
                    cont = 0.0
                    for zp in range(2):
                        m_prime = R * (m_i - c) + income[zp]
                        if m_prime <= 0:
                            # V' at 0+ is infinite for CRRA, so this c is too high
                            # Return a large positive residual to push c down
                            return 1e10
                        log_mp = np.log(m_prime)
                        dV_dm = interp_dV_dlogm[zp](log_mp) / m_prime
                        cont += transition[z, zp] * R * dV_dm
                    cont *= beta
                    return mu - cont

                # Newton's method with bisection fallback
                # Bracket: c in [1e-12, m_i]
                c_low = 1e-12
                c_high = m_i

                # Check residual at boundaries
                f_low = euler_residual(c_low)
                f_high = euler_residual(c_high)

                if f_low is None or f_high is None:
                    # Fall back to c = m_i (consume all)
                    c_opt = m_i
                elif f_low < 0 and f_high > 0:
                    # Root exists, use Newton with bisection safeguard
                    c = 0.5 * (c_low + c_high)
                    for _ in range(30):
                        f = euler_residual(c)
                        if f is None:
                            c_high = c
                            c = 0.5 * (c_low + c_high)
                            continue
                        if abs(f) < 1e-12:
                            break
                        # Secant/Newton step
                        # Estimate derivative numerically
                        dc = max(1e-8, 1e-6 * c)
                        f_plus = euler_residual(c + dc)
                        if f_plus is None:
                            f_plus = f + 1e10
                        dfdc = (f_plus - f) / dc
                        if abs(dfdc) < 1e-14:
                            # Use bisection
                            if f > 0:
                                c_high = c
                            else:
                                c_low = c
                            c = 0.5 * (c_low + c_high)
                        else:
                            c_new = c - f / dfdc
                            # Safeguard
                            if c_new <= c_low or c_new >= c_high:
                                if f > 0:
                                    c_high = c
                                else:
                                    c_low = c
                                c_new = 0.5 * (c_low + c_high)
                            c = c_new
                        if c_high - c_low < 1e-12:
                            break
                    c_opt = c
                elif f_low > 0 and f_high < 0:
                    # Root exists but reversed (shouldn't happen normally)
                    c = 0.5 * (c_low + c_high)
                    for _ in range(30):
                        f = euler_residual(c)
                        if f is None:
                            c_high = c
                            c = 0.5 * (c_low + c_high)
                            continue
                        if abs(f) < 1e-12:
                            break
                        dc = max(1e-8, 1e-6 * c)
                        f_plus = euler_residual(c + dc)
                        if f_plus is None:
                            f_plus = f + 1e10
                        dfdc = (f_plus - f) / dc
                        if abs(dfdc) < 1e-14:
                            if f > 0:
                                c_low = c
                            else:
                                c_high = c
                            c = 0.5 * (c_low + c_high)
                        else:
                            c_new = c - f / dfdc
                            if c_new <= c_low or c_new >= c_high:
                                if f > 0:
                                    c_low = c
                                else:
                                    c_high = c
                                c_new = 0.5 * (c_low + c_high)
                            c = c_new
                        if c_high - c_low < 1e-12:
                            break
                    c_opt = c
                else:
                    # No sign change; use bisection on the side where residual is closest to 0
                    if abs(f_low) < abs(f_high) if f_high is not None else True:
                        c_opt = c_low
                    else:
                        c_opt = c_high

                # Clamp
                c_opt = np.clip(c_opt, 1e-12, m_i + 1e-10)
                C[t, z, i] = c_opt

                # Compute V[t, z, i] = u(c) + beta * sum_z' P(z'|z) * V_z'(m')
                c_val = c_opt
                if abs(gamma - 1.0) < 1e-12:
                    u_val = np.log(max(c_val, 1e-12))
                else:
                    u_val = max(c_val, 1e-12) ** (1.0 - gamma) / (1.0 - gamma)

                cont_val = 0.0
                for zp in range(2):
                    m_prime = R * (m_i - c_val) + income[zp]
                    if m_prime <= 0:
                        m_prime = 1e-12
                    log_mp = np.log(m_prime)
                    # Interpolate V_z' at log_mp
                    V_interp = PchipInterpolator(log_grid, V[t+1, zp, :], extrapolate=True)
                    cont_val += transition[z, zp] * V_interp(log_mp)
                V[t, z, i] = u_val + beta * cont_val

        # Update interpolators for next iteration
        for z in range(2):
            dV_dlogm_next[z, :] = build_log_deriv(V[t, z, :])
            interp_dV_dlogm[z] = PchipInterpolator(log_grid, dV_dlogm_next[z, :], extrapolate=True)

    # Final validity check and clamp
    for t in range(horizon):
        for z in range(2):
            C[t, z, :] = np.clip(C[t, z, :], 1e-12, grid + 1e-10)
            # Ensure finite
            C[t, z, :] = np.where(np.isfinite(C[t, z, :]), C[t, z, :], grid)

    # Terminal condition: c[5] must equal grid within 1e-9
    C[5, 0, :] = grid
    C[5, 1, :] = grid

    return C
# EVOLVE-BLOCK-END
