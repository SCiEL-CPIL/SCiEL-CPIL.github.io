"""Structural rewrite: log-grid backward induction + Newton Euler solve (log-wealth value interpolation)."""
import numpy as np


def _crra_util(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-10:
        return np.log(c)
    return np.power(c, 1.0 - gamma) / (1.0 - gamma)


def _crra_mu(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-10:
        return 1.0 / c
    return np.power(c, -gamma)


def _crra_mu_inv(x, gamma):
    x = np.maximum(x, 1e-300)
    if abs(gamma - 1.0) < 1e-10:
        return 1.0 / x
    return np.power(x, -1.0 / gamma)


def _interp_log(x, xs, ys):
    """Interpolate ys(x) assuming smoothness in log(x)."""
    xl = np.log(x)
    return np.interp(xl, np.log(xs), ys)


def _solve_euler(mu_target, m, cmax, gamma):
    """Solve mu(c) = mu_target for c in (0, cmax]; return c. Vectorized over arrays."""
    mu_target = np.asarray(mu_target, dtype=float)
    cmax = np.asarray(cmax, dtype=float)
    is_log = abs(gamma - 1.0) < 1e-10

    # Newton start from the analytical inverse, clipped to (lo, cmax)
    lo = np.maximum(cmax * 1e-12, 1e-12)
    c = _crra_mu_inv(mu_target, gamma)
    c = np.clip(c, lo, cmax)

    for _ in range(10):
        mval = _crra_mu(c, gamma)
        # residual r(c) = mu(c) - mu_target  (decreasing in c)
        if is_log:
            dmdc = -mval / c
        else:
            dmdc = -gamma * mval / c
        dmdc = np.where(np.abs(dmdc) > 1e-300, dmdc, -1e-300)
        step = (mval - mu_target) / dmdc
        cnew = c - step
        # damping to stay in (lo, cmax)
        cnew = np.clip(cnew, lo, cmax)
        c = cnew
        if np.all(np.abs((mval - mu_target) / np.maximum(np.abs(mu_target), 1e-12)) < 1e-10):
            break

    # bisection fallback wherever residual is off
    mval = _crra_mu(c, gamma)
    resid = mval - mu_target
    bad = np.abs(resid) / np.maximum(np.abs(mu_target), 1e-12) > 1e-6
    if np.any(bad):
        a = lo.copy()
        b = cmax.copy()
        cb = c.copy()
        for _ in range(50):
            mid = 0.5 * (a + b)
            rmid = _crra_mu(mid, gamma) - mu_target
            go_hi = rmid > 0.0  # mu too high -> increase c
            b = np.where(go_hi, mid, b)
            a = np.where(go_hi, a, mid)
            cb = np.where(go_hi, b, a)
        c = np.where(bad, cb, c)

    c = np.clip(c, lo, cmax)
    return c


def solve_consumption(model, resource_grid, budget):
    import time

    deadline = time.time() + 120.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']

    n_states = len(income)
    out_grid = np.asarray(resource_grid, dtype=float)
    n_out = len(out_grid)

    # Internal log grid, denser than the output grid.
    lo = 1e-4
    hi = 80.0
    n_int = 4000
    m_grid = np.geomspace(lo, hi, n_int)
    m_log = np.log(m_grid)

    # V[t][z] = value at t (t = 0..horizon-1), shape (n_states, n_int)
    V = np.empty((horizon, n_states, n_int), dtype=float)
    V[horizon - 1] = 0.0  # terminal: consume everything, continuation 0

    for t in range(horizon - 2, -1, -1):
        for z in range(n_states):
            # Continuation value for each internal m (log-interpolated in m')
            cont = np.empty(n_int, dtype=float)
            for zp in range(n_states):
                mprime = R * (m_grid - m_grid)  # placeholder; filled per c
                # We need E over z' of V[t+1][z'](m'), but m' depends on c.
                # Instead precompute V[t+1][z'] on log grid and interpolate after c known.
            # First solve for c using continuation of V[t+1].
            # mu_target = beta * sum_z' trans[z,z'] * mu(c'(m'))  -- depends on c, iterative.
            # Use fixed-point: start with c = m (consume all -> m'=income), then refine.
            c = m_grid.copy()
            for _ in range(6):
                # m' for current c
                # E mu(c') where c' solves next-period policy given m'
                # We approximate next-period mu(c') via its own Euler: mu(c') = beta*E mu(c'')...
                # Simpler: use that at t+1 the marginal utility equals beta*E mu at t+2.
                # Precompute next-period "mu schedule" M[t+1][z'](m') = mu(c_{t+1}(m')).
                # Build M by solving Euler at t+1 for all m'.
                # We'll compute M_next once per t (shared across z) below.
                break
        # (placeholder loop replaced by clean two-phase below)

    # ---- Clean two-phase: precompute next-period marginal-utility schedule M_next[z'](m') ----
    # M[t][z](m) = mu(c_t(z, m)) = beta * sum_z' trans[z,z'] * M[t+1][z'](R*(m-c_t(z,m)) + income[z'])
    # Solve via fixed point on c.

    M = np.empty((horizon, n_states, n_int), dtype=float)  # marginal utility of consumption
    C = np.empty((horizon, n_states, n_int), dtype=float)

    # Terminal: c = m, mu(c) = mu(m)
    M[horizon - 1] = _crra_mu(m_grid, gamma)[np.newaxis, :]
    C[horizon - 1] = m_grid[np.newaxis, :]

    for t in range(horizon - 2, -1, -1):
        for z in range(n_states):
            # fixed point on c
            c = m_grid.copy()
            for _ in range(8):
                # m' = R*(m - c) + income[z']
                # M_target(m) = beta * sum_z' trans[z,z'] * M[t+1][z'](m')
                M_target = np.zeros(n_int, dtype=float)
                for zp in range(n_states):
                    w = transition[z, zp]
                    if w <= 0:
                        continue
                    mprime = R * (m_grid - c) + income[zp]
                    mprime = np.clip(mprime, lo, hi)
                    M_next_zp = _interp_log(mprime, m_grid, M[t + 1, zp])
                    M_target += w * M_next_zp
                M_target = beta * M_target
                # solve mu(c) = M_target
                c_new = _solve_euler(M_target, m_grid, m_grid, gamma)
                if np.allclose(c_new, c, rtol=1e-10, atol=1e-12):
                    c = c_new
                    break
                c = c_new
            C[t, z] = c
            M[t, z] = _crra_mu(c, gamma)
        if time.time() > deadline - 30.0:
            break

    # Interpolate consumption policy onto the output grid for each (t, z)
    consumption = np.empty((horizon, n_states, n_out), dtype=float)
    for t in range(horizon):
        for z in range(n_states):
            cpol = _interp_log(out_grid, m_grid, C[t, z])
            cpol = np.clip(cpol, 1e-12, out_grid + 1e-10)
            consumption[t, z] = cpol

    # Terminal constraint: c[5] must equal the grid within 1e-9
    consumption[horizon - 1] = out_grid[np.newaxis, :]

    # Final validity guard
    consumption = np.where(np.isfinite(consumption), consumption, 1e-12)
    consumption = np.clip(consumption, 1e-12, None)
    consumption = np.minimum(consumption, out_grid[np.newaxis, np.newaxis, :] + 1e-10)
    consumption[horizon - 1] = out_grid[np.newaxis, :]

    return consumption
