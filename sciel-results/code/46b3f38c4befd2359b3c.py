"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import make_interp_spline


def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    n_z = len(income)

    m_grid = np.geomspace(0.001, 40, 251)
    n_m = len(m_grid)

    def u(c):
        cs = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(cs)
        return cs ** (1.0 - gamma) / (1.0 - gamma)

    def up(c):
        cs = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / cs
        return cs ** (-gamma)

    def upp(c):
        cs = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return -1.0 / (cs * cs)
        return gamma * cs ** (-gamma - 1.0)

    c_policy = np.zeros((horizon, n_z, n_m))
    V = np.zeros((horizon, n_z, n_m))
    for z in range(n_z):
        c_policy[horizon - 1, z, :] = m_grid
        V[horizon - 1, z, :] = u(m_grid)

    # Vectorized second-difference operator
    dm = np.diff(m_grid)
    dm_mid = 0.5 * (m_grid[1:] + m_grid[:-1])
    W1 = np.zeros((n_m, n_m)); W1[np.arange(n_m - 1), np.arange(1, n_m)] = 1.0 / (2.0 * dm_mid)
    W2 = np.zeros((n_m, n_m)); W2[np.arange(n_m - 1), np.arange(1, n_m)] = -1.0 / dm
    W3 = np.zeros((n_m, n_m)); W3[np.arange(1, n_m), np.arange(n_m - 1)] = 1.0 / dm
    W4 = np.zeros((n_m, n_m)); W4[np.arange(1, n_m), np.arange(n_m - 1)] = 1.0 / (2.0 * dm_mid)
    W = W1 + W2 + W3 + W4  # V''(m_i) = sum_j W[i,j] V(m_j)

    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            break
        h = 0.5 * (m_grid[1:] + m_grid[:-1])

        # Build splines for V[t+1, z] (skip for last backward step)
        if t < horizon - 3:
            v_splines = []
            for zp in range(n_z):
                try:
                    v_splines.append(make_interp_spline(m_grid, V[t + 1, zp, :], k=3, s=0))
                except Exception:
                    v_splines.append(None)
        else:
            v_splines = [None] * n_z

        # Vectorized V''[t+1, z] = V_{t+1} @ W.T
        Vpp = np.zeros((n_z, n_m))
        for zp in range(n_z):
            Vpp[zp, :] = V[t + 1, zp, :] @ W.T

        for z in range(n_z):
            P = transition[z]
            # Effective continuation curvature (sum over z' of P[z,z'] * V''[t+1,z'])
            Vpp_eff = P @ Vpp  # shape (n_m,)

            c = np.empty(n_m)
            Vt = np.empty(n_m)
            for i in range(n_m):
                m_i = m_grid[i]
                if m_i < 1e-8:
                    c[i] = m_i
                    Vt[i] = u(m_i)
                    continue
                c_lo = max(m_i * 1e-8, 1e-12)
                c_hi = m_i * (1.0 - 1e-10)
                if c_hi <= c_lo:
                    c[i] = m_i
                    Vt[i] = u(m_i)
                    continue
                # Initial guess: previous-period policy or 50%
                if t == horizon - 3:
                    c0 = m_i * 0.5
                else:
                    c0 = c_policy[t + 1, z, i]
                c0 = min(max(c0, c_lo * 1.0001), c_hi * 0.9999)

                for _ in range(25):
                    x = R * (m_i - c0)
                    # Evaluate continuation value and derivative at x for each z'
                    if v_splines[0] is not None:
                        vals = np.empty(n_z)
                        dvals = np.empty(n_z)
                        for zp in range(n_z):
                            if P[zp] < 1e-15:
                                vals[zp] = 0.0; dvals[zp] = 0.0
                                continue
                            mzp = x + income[zp]
                            mzp = min(max(mzp, 1e-12), 40.0)
                            vals[zp] = v_splines[zp](mzp)
                            dvals[zp] = v_splines[zp](mzp, 1)
                        f = up(c0) - beta * (P @ dvals)
                        cont_v = P @ vals
                        # Derivative: -u''(c0) - beta * R * Vpp_eff(x) (approx, x is near m_grid)
                        # Need Vpp_eff evaluated at x (not at a grid point); use local interpolation
                        # For simplicity use nearest grid index
                        ix = int(np.clip(np.searchsorted(m_grid, x), 1, n_m - 2))
                        Vpp_at_x = Vpp_eff[ix]
                        dfdc = -upp(c0) - beta * R * Vpp_at_x
                        if abs(dfdc) < 1e-15:
                            break
                        step = f / dfdc
                        if abs(step) > 0.5 * c0:
                            step = np.sign(step) * 0.5 * c0
                        c1 = c0 + step
                        if c1 <= c_lo:
                            c1 = c_lo * 1.0001
                        elif c1 >= c_hi:
                            c1 = c_hi * 0.9999
                        if abs(c1 - c0) < 1e-10 * max(1.0, c0):
                            c0 = c1
                            break
                        c0 = c1
                    else:
                        # Linear interp fallback
                        vals = np.empty(n_z)
                        dvals = np.empty(n_z)
                        for zp in range(n_z):
                            if P[zp] < 1e-15:
                                vals[zp] = 0.0; dvals[zp] = 0.0
                                continue
                            mzp = x + income[zp]
                            mzp = min(max(mzp, 1e-12), 40.0)
                            vals[zp] = float(np.interp(mzp, m_grid, V[t + 1, zp, :]))
                            dvals[zp] = float(np.interp(mzp, m_grid, np.concatenate(([0.0], np.diff(V[t + 1, zp, :]) / np.diff(m_grid), [0.0]))))
                        f = up(c0) - beta * (P @ dvals)
                        cont_v = P @ vals
                        Vpp_at_x = Vpp_eff[int(np.clip(np.searchsorted(m_grid, x), 0, n_m - 1))]
                        dfdc = -upp(c0) - beta * R * Vpp_at_x
                        if abs(dfdc) < 1e-15:
                            break
                        step = f / dfdc
                        if abs(step) > 0.5 * c0:
                            step = np.sign(step) * 0.5 * c0
                        c1 = c0 + step
                        if c1 <= c_lo:
                            c1 = c_lo * 1.0001
                        elif c1 >= c_hi:
                            c1 = c_hi * 0.9999
                        if abs(c1 - c0) < 1e-10 * max(1.0, c0):
                            c0 = c1
                            break
                        c0 = c1

                c0 = min(max(c0, c_lo), c_hi)
                c[i] = c0
                # Compute V[t,z,i]
                x = R * (m_i - c0)
                cont = 0.0
                for zp in range(n_z):
                    if P[zp] < 1e-15:
                        continue
                    mzp = x + income[zp]
                    mzp = min(max(mzp, 1e-12), 40.0)
                    if v_splines[zp] is not None:
                        cont += P[zp] * v_splines[zp](mzp)
                    else:
                        cont += P[zp] * float(np.interp(mzp, m_grid, V[t + 1, zp, :]))
                Vt[i] = u(c0) + beta * cont

            c_policy[t, z, :] = c
            V[t, z, :] = Vt

    # Interpolate to resource_grid
    result = np.zeros((horizon, n_z, len(resource_grid)))
    for t in range(horizon):
        for z in range(n_z):
            result[t, z, :] = np.interp(resource_grid, m_grid, c_policy[t, z, :])

    # Enforce constraints
    for t in range(horizon):
        for z in range(n_z):
            result[t, z, :] = np.clip(result[t, z, :], 1e-12, resource_grid + 1e-10)
    for z in range(n_z):
        result[horizon - 1, z, :] = resource_grid

    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=1e-12, posinf=1.0, neginf=1e-12)
        for t in range(horizon):
            for z in range(n_z):
                result[t, z, :] = np.clip(result[t, z, :], 1e-12, resource_grid + 1e-10)
        for z in range(n_z):
            result[horizon - 1, z, :] = resource_grid

    return result
# EVOLVE-BLOCK-END
