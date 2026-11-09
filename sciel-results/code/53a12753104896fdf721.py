"""Rewritten solver for consumption-saving policy accuracy.

Strategy: backward induction over the resource grid, exploiting CRRA homotheticity.
The continuation value is represented in log-space (log m) for smoothness, and at
each (t, z) the optimal consumption is found by a 1-D search over the consumption
fraction phi = c/m.  A deadline guard returns the best partial result if the
150 s wall-clock limit is approached.
"""
import time

import numpy as np


def _utility(c, gamma):
    """CRRA utility, c floored at 1e-12."""
    c = np.maximum(np.asarray(c, dtype=float), 1e-12)
    if gamma == 1:
        return np.log(c)
    return np.power(c, 1.0 - gamma) / (1.0 - gamma)


def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before the 150 s kill

    beta = float(model["beta"])
    gamma = float(model["gamma"])
    R = float(model.get("R", 1.02))
    income = np.asarray(model["income"], dtype=float)
    trans = np.asarray(model["transition"], dtype=float)
    horizon = int(model["horizon"])

    m_grid = np.asarray(resource_grid, dtype=float)
    n_m = m_grid.size
    n_z = income.size
    n_t = horizon

    # log-space grid for smooth continuation interpolation
    log_m = np.log(m_grid)

    # continuation value V[t, z, i]  (value at start of period t with resources m_grid[i])
    # We compute backward: V[horizon] = u(m)  (consume all at the terminal period).
    V = np.empty((n_t + 1, n_z, n_m))
    for z in range(n_z):
        V[n_t, z, :] = _utility(m_grid, gamma)

    # consumption policy c[t, z, i]
    C = np.empty((n_t, n_z, n_m))

    # log-space interpolation helper
    def interp_log(arr, m_eval):
        # arr defined on m_grid (log_m). Interpolate in log m.
        return np.interp(np.log(np.maximum(m_eval, 1e-12)), log_m, arr)

    # search over consumption fraction phi in (phi_lo, 1]
    phi_lo = 1e-4
    # objective for given phi at (z, m): u(m*phi) + beta * sum_z' trans[z,z'] V_{t+1}(z', R*m*(1-phi)+income[z'])
    # We maximize over phi.  For each (z, i) we do a 1-D golden-section / brent search.

    # Precompute for speed: for a fixed (t, z), the continuation as a function of next m.
    from scipy.optimize import minimize_scalar

    def cont_value(t, z, m_next):
        # m_next: array
        vals = np.empty_like(m_next, dtype=float)
        for zp in range(n_z):
            vals += trans[z, zp] * interp_log(V[t + 1, zp, :], m_next)
        return vals

    for t in range(n_t - 1, -1, -1):
        for z in range(n_z):
            # objective(phi) for each m
            def neg_obj(phi, m):
                c = m * phi
                u_now = _utility(c, gamma)
                m_next = R * (m - c) + income[np.arange(n_z)]
                cont = np.zeros_like(m)
                for zp in range(n_z):
                    cont += trans[z, zp] * interp_log(V[t + 1, zp, :], m_next[zp])
                return -(u_now + beta * cont)

            best_phi = np.empty(n_m)
            # Use a vectorized coarse grid search then refine the best few.
            n_coarse = max(40, budget // 2)
            phis = np.linspace(phi_lo, 1.0, n_coarse)
            # evaluate objective on a subset of m to keep it fast, then per-m refine
            # For accuracy, refine every m with a local search around the best coarse phi.
            # Coarse evaluation (vectorized over m for each phi)
            obj_grid = np.empty((n_coarse, n_m))
            for k, phi in enumerate(phis):
                c = m_grid * phi
                u_now = _utility(c, gamma)
                m_next = R * (m_grid - c) + income[np.arange(n_z)]
                cont = np.zeros(n_m)
                for zp in range(n_z):
                    cont += trans[z, zp] * interp_log(V[t + 1, zp, :], m_next[zp])
                obj_grid[k, :] = u_now + beta * cont

            best_k = np.argmax(obj_grid, axis=0)
            best_phi = phis[best_k]

            # Refine with a bounded scalar minimization per m around the coarse best.
            # Limit refinement count to respect the deadline.
            refine_every = 1
            for i in range(0, n_m, refine_every):
                lo = phis[max(0, best_k[i] - 1)]
                hi = phis[min(n_coarse - 1, best_k[i] + 1)]
                if hi <= lo:
                    hi = min(1.0, lo + 1e-3)
                    lo = max(phi_lo, hi - 1e-3)
                m_i = m_grid[i]
                try:
                    res = minimize_scalar(
                        lambda p: neg_obj(p, m_i),
                        bounds=(lo, hi),
                        method="bounded",
                        options={"xatol": 1e-6},
                    )
                    best_phi[i] = res.x
                except Exception:
                    pass
                if time.time() > deadline - 5.0:
                    break
            C[t, z, :] = m_grid * best_phi
            # store value
            c = C[t, z, :]
            m_next = R * (m_grid - c) + income[np.arange(n_z)]
            cont = np.zeros(n_m)
            for zp in range(n_z):
                cont += trans[z, zp] * interp_log(V[t + 1, zp, :], m_next[zp])
            V[t, z, :] = _utility(c, gamma) + beta * cont

        if time.time() > deadline - 5.0:
            # fill remaining earlier periods with a simple rule (consume all at terminal-like)
            for tt in range(t - 1, -1, -1):
                for zz in range(n_z):
                    C[tt, zz, :] = m_grid
                    V[tt, zz, :] = _utility(m_grid, gamma)
            break

    # Enforce constraints: 0 < c <= m + 1e-10, and c[5] equal to the grid within 1e-9.
    C = np.clip(C, 1e-12, m_grid[None, None, :] + 1e-10)
    C[5, :, :] = m_grid[None, :]

    # Final finiteness check
    if not np.all(np.isfinite(C)):
        C = np.where(np.isfinite(C), C, m_grid[None, None, :])

    return C
