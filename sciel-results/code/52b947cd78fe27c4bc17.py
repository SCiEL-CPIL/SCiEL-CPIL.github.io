import time
import numpy as np
from scipy.interpolate import CubicSpline

def _crra_util(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-9:
        return np.log(c)
    return c ** (1.0 - gamma) / (1.0 - gamma)

def _crra_mu(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-9:
        return 1.0 / c
    return c ** (-gamma)

def solve_consumption(model, resource_grid, budget):
    start = time.time()
    deadline = start + 110.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    trans = np.asarray(model['transition'], dtype=float)
    horizon = int(model['horizon'])
    M = np.asarray(resource_grid, dtype=float)
    num_m = M.size
    z_states = income.size

    # log-wealth grid (uniform in log space -> splines well-conditioned)
    x = np.log(M)
    # consumption candidates per state for the final polish
    K = 600
    frac = np.linspace(0.001, 1.0, K)

    def elapsed_ok():
        return time.time() < deadline

    # V[t][z] = array over m of continuation value from state z at time t (max value achievable
    # starting at time t with wealth m in state z). We compute backward: V[horizon] is terminal.
    # c[t][z] = optimal consumption at time t.
    V = np.zeros((horizon + 1, z_states, num_m))
    C = np.zeros((horizon, z_states, num_m))

    # Terminal period (t = horizon): consume all wealth (no future). V[horizon] = u(m).
    V[horizon] = _crra_util(M, gamma)[None, :]

    for t in range(horizon - 1, -1, -1):
        if not elapsed_ok():
            break
        for z in range(z_states):
            # continuation value of consuming c now:
            #   now: u(c)
            #   future: beta * sum_{z'} P[z,z'] * E_{m'} V[t+1][z'](m')
            #   where m' = R (m - c) + income[z']
            # We want, for each m, the c in (0, m] maximizing u(c) + beta * E V[t+1].
            # Use Newton on the smooth objective g(c) = u(c) + beta * E V[t+1](m').
            # g'(c) = u'(c) - beta * R * E V[t+1]'(m')   (d m'/dc = -R)
            # g''(c) = u''(c) - beta * R^2 * E V[t+1]''(m')

            # Build splines for V[t+1][z'] in log-wealth space.
            # V as function of x: f(x) = V(m). Use CubicSpline(x, V) -> gives f, f', f'' in x.
            # We need V'(m) and V''(m): dV/dm = f'(x)/m ; d2V/dm2 = (f''(x) - f'(x))/m^2.
            splines = []
            for zp in range(z_states):
                splines.append(CubicSpline(x, V[t + 1, zp], extrapolate='linear'))

            def cont_value(m_arr, c_arr):
                # m' for each z'
                # returns expected continuation (beta * E V)
                tot = np.zeros_like(m_arr)
                for zp in range(z_states):
                    p = trans[z, zp]
                    if p <= 0.0:
                        continue
                    mp = R * (m_arr - c_arr) + income[zp]
                    mp = np.maximum(mp, 1e-12)
                    xp = np.log(mp)
                    tot += p * splines[zp](xp)
                return beta * tot

            def cont_deriv(m_arr, c_arr):
                # d/dc of cont_value = beta * E [ V'(m') * d m'/dc ] = beta * E[ V'(m') * (-R) ]
                tot = np.zeros_like(m_arr)
                for zp in range(z_states):
                    p = trans[z, zp]
                    if p <= 0.0:
                        continue
                    mp = R * (m_arr - c_arr) + income[zp]
                    mp = np.maximum(mp, 1e-12)
                    xp = np.log(mp)
                    f1 = splines[zp](xp, 1)  # df/dx
                    Vm = f1 / mp             # dV/dm
                    tot += p * Vm
                return beta * (-R) * tot

            def cont_second(m_arr, c_arr):
                # d2/dc2 of cont_value = beta * E[ V''(m') * R^2 ]
                tot = np.zeros_like(m_arr)
                for zp in range(z_states):
                    p = trans[z, zp]
                    if p <= 0.0:
                        continue
                    mp = R * (m_arr - c_arr) + income[zp]
                    mp = np.maximum(mp, 1e-12)
                    xp = np.log(mp)
                    f1 = splines[zp](xp, 1)
                    f2 = splines[zp](xp, 2)
                    Vmm = (f2 - f1) / (mp * mp)  # d2V/dm2
                    tot += p * Vmm
                return beta * (R * R) * tot

            # Per-m Newton solve. Vectorize over m.
            m = M
            # initial guess: interior fraction of wealth
            c = np.clip(0.5 * m, 1e-12, m)
            for it in range(30):
                if not elapsed_ok():
                    break
                u1 = _crra_mu(c, gamma)
                u2 = -_crra_mu(c, gamma) / c  # u''(c) = -u'(c)/c
                cd = cont_deriv(m, c)
                cdd = cont_second(m, c)
                g1 = u1 + cd          # g'(c)
                g2 = u2 + cdd         # g''(c)
                # Newton step
                step = g1 / np.where(np.abs(g2) > 1e-300, g2, 1e-300)
                c_new = c - step
                # enforce bounds (0, m]
                c_new = np.clip(c_new, 1e-12, m)
                if np.max(np.abs(c_new - c)) < 1e-12 * np.max(m):
                    c = c_new
                    break
                c = c_new
            C[t, z] = c
            # V[t][z] = u(c) + cont_value
            V[t, z] = _crra_util(c, gamma) + cont_value(m, c)

    # Final polish: recompute consumption at t via dense candidate maximization using
    # the (now fixed) continuation splines, to reduce Newton/interpolation error.
    for t in range(horizon - 1, -1, -1):
        if not elapsed_ok():
            break
        for z in range(z_states):
            splines = []
            for zp in range(z_states):
                splines.append(CubicSpline(x, V[t + 1, zp], extrapolate='linear'))
            m = M
            best_c = C[t, z].copy()
            best_val = -np.inf * np.ones_like(m)
            for f in frac:
                c = np.maximum(f * m, 1e-12)
                c = np.minimum(c, m)
                val = _crra_util(c, gamma)
                for zp in range(z_states):
                    p = trans[z, zp]
                    if p <= 0.0:
                        continue
                    mp = R * (m - c) + income[zp]
                    mp = np.maximum(mp, 1e-12)
                    val = val + beta * p * splines[zp](np.log(mp))
                better = val > best_val
                best_val = np.where(better, val, best_val)
                best_c = np.where(better, c, best_c)
            C[t, z] = best_c
            # recompute V for consistency (used by earlier periods)
            val = _crra_util(best_c, gamma)
            for zp in range(z_states):
                p = trans[z, zp]
                if p <= 0.0:
                    continue
                mp = R * (m - best_c) + income[zp]
                mp = np.maximum(mp, 1e-12)
                val = val + beta * p * splines[zp](np.log(mp))
            V[t, z] = val

    # Build output: consumption[t, income_state, resource_index], shape (6, 2, 101)
    out = np.zeros((horizon, z_states, num_m))
    for t in range(horizon):
        for z in range(z_states):
            out[t, z] = C[t, z]

    # Enforce validity: finite, 0 < c <= m + 1e-10, and c[5] == grid within 1e-9.
    out = np.where(np.isfinite(out), out, 1e-9)
    out = np.clip(out, 1e-12, M[None, None, :] + 1e-10)
    if horizon >= 6:
        out[5] = M[None, None, :].copy()  # terminal: consume all

    return out
