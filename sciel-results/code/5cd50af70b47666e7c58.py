"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    grid = np.asarray(resource_grid, dtype=float)
    n_grid = len(grid)
    n_z = len(income)

    is_log = abs(gamma - 1.0) < 1e-10

    def u(c):
        c = np.maximum(c, 1e-12)
        if is_log:
            return np.log(c)
        return c ** (1.0 - gamma) / (1.0 - gamma)

    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if is_log:
            return 1.0 / c
        return c ** (-gamma)

    log_grid = np.log(grid)

    # V[t, z, i] value function; C[t, z, i] consumption policy
    V = np.zeros((horizon, n_z, n_grid))
    C = np.zeros((horizon, n_z, n_grid))

    # Terminal period: consume everything
    C[horizon - 1, :, :] = grid[np.newaxis, np.newaxis, :]
    V[horizon - 1, :, :] = u(grid)[np.newaxis, np.newaxis, :]

    # Continuation value W[t, z, i] = E_{z'}[V[t+1, z', R*(m-c) + income[z']]]
    # Wp = derivative of W w.r.t. m = R * E_{z'}[Vp[t+1, z', ...]]
    # Store Wp as log-derivative d(log W)/d(log m) = Wp * m / W  (smooth for FOC)

    Wp = np.zeros((horizon, n_z, n_grid))
    logWp = np.zeros((horizon, n_z, n_grid))

    # Terminal continuation: from horizon-1 onward nothing (V=0 beyond)
    # Actually at t = horizon-2, continuation is V[horizon-1] = u(m')
    # We'll compute W and Wp for each t on the fly.

    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break

        # Build interpolation tables for V[t+1] and Vp[t+1] in log-space
        # Vp[t+1] = u'(C[t+1]) by envelope theorem
        Vp_next = u_prime(np.maximum(C[t + 1], 1e-12))  # (n_z, n_grid)
        V_next = V[t + 1]  # (n_z, n_grid)

        # Precompute for each z' the log-space interpolation arrays
        # We'll interpolate in log_m for both V and Vp.

        for z in range(n_z):
            c_opt = np.zeros(n_grid)
            for i in range(n_grid):
                m = grid[i]
                if m < 1e-10:
                    c_opt[i] = 1e-12
                    continue

                # Coarse ratio search to bracket the FOC root
                # f in [0.001, 1], c = f * m
                n_ratio = 40
                ratios = np.linspace(0.001, 1.0, n_ratio)

                def foc_residual(c):
                    c_safe = max(c, 1e-12)
                    lhs = u_prime(c_safe)
                    rhs = 0.0
                    for zp in range(n_z):
                        m_prime = R * (m - c_safe) + income[zp]
                        if m_prime < 1e-12:
                            m_prime = 1e-12
                        log_mp = np.log(m_prime)
                        if log_mp <= log_grid[0]:
                            vp = Vp_next[zp, 0]
                        elif log_mp >= log_grid[-1]:
                            vp = Vp_next[zp, -1]
                        else:
                            vp = np.interp(log_mp, log_grid, Vp_next[zp, :])
                        rhs += transition[z, zp] * vp
                    return lhs - beta * R * rhs

                # Coarse scan to find sign change
                f_prev = None
                c_prev = None
                c_lo = 1e-12
                c_hi = m
                found = False
                for ratio in ratios:
                    c = ratio * m
                    if c >= m:
                        c = m
                    c = min(max(c, 1e-12), m)
                    f_val = foc_residual(c)
                    if f_prev is not None:
                        if f_prev * f_val <= 0:
                            # sign change between c_prev and c
                            a, b = c_prev, c
                            try:
                                c_opt[i] = brentq(foc_residual, a, b, xtol=1e-12, rtol=1e-10)
                            except Exception:
                                # bisection fallback
                                lo, hi = a, b
                                for _ in range(60):
                                    mid = 0.5 * (lo + hi)
                                    fm = foc_residual(mid)
                                    if fm < 0:
                                        lo = mid
                                    else:
                                        hi = mid
                                c_opt[i] = 0.5 * (lo + hi)
                            found = True
                            break
                    f_prev = f_val
                    c_prev = c

                if not found:
                    # No sign change: check endpoints
                    f_lo = foc_residual(c_lo)
                    f_hi = foc_residual(c_hi)
                    if f_lo < 0:
                        c_opt[i] = c_lo
                    elif f_hi > 0:
                        c_opt[i] = c_hi
                    else:
                        # Fallback: pick best on a finer ratio grid
                        best_c = c_lo
                        best_val = -np.inf
                        fine_ratios = np.linspace(0.001, 1.0, 200)
                        for ratio in fine_ratios:
                            c = min(max(ratio * m, 1e-12), m)
                            # compute total value
                            cont = 0.0
                            for zp in range(n_z):
                                m_prime = R * (m - c) + income[zp]
                                if m_prime < 1e-12:
                                    m_prime = 1e-12
                                log_mp = np.log(m_prime)
                                if log_mp <= log_grid[0]:
                                    vv = V_next[zp, 0]
                                elif log_mp >= log_grid[-1]:
                                    vv = V_next[zp, -1]
                                else:
                                    vv = np.interp(log_mp, log_grid, V_next[zp, :])
                                cont += transition[z, zp] * vv
                            total = u(c) + beta * cont
                            if total > best_val:
                                best_val = total
                                best_c = c
                        c_opt[i] = best_c

                c_opt[i] = np.clip(c_opt[i], 1e-12, m)

            C[t, z, :] = c_opt
            # Value: u(c) + beta * E[V_next]
            for i in range(n_grid):
                m = grid[i]
                c = c_opt[i]
                cont = 0.0
                for zp in range(n_z):
                    m_prime = R * (m - c) + income[zp]
                    if m_prime < 1e-12:
                        m_prime = 1e-12
                    log_mp = np.log(m_prime)
                    if log_mp <= log_grid[0]:
                        vv = V_next[zp, 0]
                    elif log_mp >= log_grid[-1]:
                        vv = V_next[zp, -1]
                    else:
                        vv = np.interp(log_mp, log_grid, V_next[zp, :])
                    cont += transition[z, zp] * vv
                V[t, z, i] = u(c) + beta * cont

    return _finalize(C, grid, horizon, n_z, n_grid)

def _finalize(C, grid, horizon, n_z, n_grid):
    # Ensure terminal condition
    C[horizon - 1, :, :] = grid[np.newaxis, np.newaxis, :]
    # Ensure 0 < c <= m + 1e-10 and finite
    for t in range(horizon):
        for z in range(n_z):
            C[t, z, :] = np.clip(C[t, z, :], 1e-12, grid + 1e-10)
    C = np.where(np.isfinite(C), C, 1e-12)
    return C
# EVOLVE-BLOCK-END
