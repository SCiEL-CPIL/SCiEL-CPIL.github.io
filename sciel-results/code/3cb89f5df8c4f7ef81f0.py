# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_TOL = 1e-7


def _pairwise_clearance(cx, cy, s):
    """Return (dx, dy, t) arrays of shape (n, n) for all pairs i<j via broadcasting.

    dx[i,j] = cx[i]-cx[j], dy[i,j] = cy[i]-cy[j], t[i,j] = (s[i]+s[j])/2
    """
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    t = 0.5 * (s[:, None] + s[None, :])
    return dx, dy, t


def _overlap_penalty_vec(cx, cy, s, beta=800.0, eps=1e-12):
    """Smooth sum of overlap violations over all pairs and walls.

    Overlap in x when |dx| < t, in y when |dy| < t. A square overlaps another
    iff both. We penalize max(0, t-|dx|) * max(0, t-|dy|) which is 0 when either
    axis is separated by >= t (touching allowed) and positive otherwise.
    """
    dx, dy, t = _pairwise_clearance(cx, cy, s)
    # only upper triangle to avoid double counting
    iu = np.triu_indices(len(cx), k=1)
    ox = np.maximum(0.0, t[iu] - np.abs(dx[iu]))
    oy = np.maximum(0.0, t[iu] - np.abs(dy[iu]))
    pair_pen = ox * oy  # zero unless both axes overlap
    # smooth: use soft product
    pair_pen = pair_pen ** 2
    pair_sum = float(np.sum(pair_pen))

    # Wall penalties: square must be within [0,1]. Half-side h = s/2.
    # left: cx - h >= 0  -> violation max(0, h - cx)
    # right: cx + h <= 1 -> violation max(0, cx + h - 1)
    h = 0.5 * s
    wl = np.maximum(0.0, h - cx)
    wr = np.maximum(0.0, cx + h - 1.0)
    wb = np.maximum(0.0, h - cy)
    wt = np.maximum(0.0, cy + h - 1.0)
    wall_pen = float(np.sum(wl**2 + wr**2 + wb**2 + wt**2))

    return pair_sum + wall_pen


def _objective_and_grad(z, s, beta=800.0):
    """Objective for fixed sides: maximize sum s (constant) - penalty.
    z = (cx, cy) length 2n. Returns (value, grad)."""
    n = len(s)
    cx = z[:n]
    cy = z[n:]

    dx, dy, t = _pairwise_clearance(cx, cy, s)
    iu = np.triu_indices(n, k=1)

    # pair part
    adx = np.abs(dx[iu])
    ady = np.abs(dy[iu])
    ti = t[iu]
    oi = np.maximum(0.0, ti - adx)
    oj = np.maximum(0.0, ti - ady)
    pair_pen = np.sum((oi * oj) ** 2)

    # wall part
    h = 0.5 * s
    wl = np.maximum(0.0, h - cx)
    wr = np.maximum(0.0, cx + h - 1.0)
    wb = np.maximum(0.0, h - cy)
    wt = np.maximum(0.0, cy + h - 1.0)
    wall_pen = np.sum(wl**2 + wr**2 + wb**2 + wt**2)

    pen = pair_pen + wall_pen

    # gradient
    grad = np.zeros(2 * n)
    # pair gradients
    # d(oi*oj)^2 / d(cx_i) ... oi depends on cx_i (i) and cx_j (j)
    # Let's compute via chain rule numerically-ish but analytically.
    # For pair (i,j): f = (oi*oj)^2
    # doi/d(cx_i) = -sign(dx) if oi>0 else 0 ; doi/d(cx_j) = +sign(dx) if oi>0 else 0
    # where dx = cx_i - cx_j, adx=|dx|
    signx = np.sign(dx[iu])
    signy = np.sign(dy[iu])
    # d(oi*oj)^2/d oi = 2*oi*oj^2 ; d(oi*oj)^2/d oj = 2*oi^2*oj
    dfdoi = 2.0 * oi * oj**2
    dfdoj = 2.0 * oi**2 * oj
    # contributions
    # For each pair (i,j) in iu:
    ii, jj = iu
    # d pen/d cx_i += dfdoi * doi/d cx_i = dfdoi * (-signx)
    grad[ii] += dfdoi * (-signx)
    grad[jj] += dfdoi * (signx)
    grad[ii + n] += dfdoj * (-signy)
    grad[jj + n] += dfdoj * (signy)

    # wall gradients
    grad[:n] += 2.0 * wl * (-1.0) + 2.0 * wr * (1.0)
    grad[n:] += 2.0 * wb * (-1.0) + 2.0 * wt * (1.0)

    # We want to MINIMIZE penalty (since sum s is constant for fixed s).
    return pen, grad


def _is_valid(centers, sides, tol=_TOL):
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    cx = centers[:, 0]
    cy = centers[:, 1]
    h = 0.5 * sides
    if np.any(cx - h < -tol) or np.any(cx + h > 1.0 + tol):
        return False
    if np.any(cy - h < -tol) or np.any(cy + h > 1.0 + tol):
        return False
    dx, dy, t = _pairwise_clearance(cx, cy, sides)
    iu = np.triu_indices(n, k=1)
    adx = np.abs(dx[iu])
    ady = np.abs(dy[iu])
    ti = t[iu]
    if np.any(adx < ti - tol) and np.any(ady < ti - tol):
        # need per-pair check
        bad = (adx < ti - tol) & (ady < ti - tol)
        if np.any(bad):
            return False
    return True


def _grid_packing():
    n = 10
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides


def _optimize_centers(s, z0, time_budget, beta=800.0):
    """Optimize centers for fixed sides. Returns best (centers, pen)."""
    n = len(s)
    best_z = z0.copy()
    best_pen = _objective_and_grad(z0, s, beta)[0]

    def fun(z):
        return _objective_and_grad(z, s, beta)[0]

    def jac(z):
        return _objective_and_grad(z, s, beta)[1]

    t0 = time.time()
    # L-BFGS-B with bounds
    bounds = [(0.0, 1.0)] * (2 * n)
    res = minimize(fun, z0, jac=jac, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
    if time.time() - t0 > time_budget:
        return None, best_pen
    if res.fun < best_pen:
        best_pen = res.fun
        best_z = res.x

    # Local polishing with Nelder-Mead from several random starts if time
    t1 = time.time()
    remaining = time_budget - (t1 - t0)
    if remaining > 0.5:
        rng = np.random.default_rng(0)
        nstarts = max(1, int(remaining * 2))
        for _ in range(nstarts):
            if time.time() - t0 > time_budget:
                break
            z_start = best_z + rng.normal(0, 0.02, 2 * n)
            z_start = np.clip(z_start, 0, 1)
            r2 = minimize(fun, z_start, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14})
            if r2.fun < best_pen:
                best_pen = r2.fun
                best_z = r2.x

    cx = best_z[:n]
    cy = best_z[n:]
    centers = np.stack([cx, cy], axis=1)
    return centers, best_pen


def _optimize_sides(centers, s0, time_budget, beta=800.0):
    """Optimize sides for fixed centers. Returns best (sides, pen)."""
    n = len(s0)
    best_s = s0.copy()
    best_pen = _overlap_penalty_vec(centers[:, 0], centers[:, 1], s0, beta)

    def fun_s(s):
        return _overlap_penalty_vec(centers[:, 0], centers[:, 1], s, beta)

    bounds = [(1e-4, 1.0)] * n
    t0 = time.time()
    res = minimize(fun_s, s0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 2000, 'ftol': 1e-14})
    if time.time() - t0 > time_budget:
        return None, best_pen
    if res.fun < best_pen:
        best_pen = res.fun
        best_s = res.x

    t1 = time.time()
    remaining = time_budget - (t1 - t0)
    if remaining > 0.5:
        rng = np.random.default_rng(1)
        nstarts = max(1, int(remaining * 2))
        for _ in range(nstarts):
            if time.time() - t0 > time_budget:
                break
            s_start = np.clip(best_s + rng.normal(0, 0.02, n), 1e-4, 1.0)
            r2 = minimize(fun_s, s_start, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14})
            if r2.fun < best_pen:
                best_pen = r2.fun
                best_s = r2.x

    return best_s, best_pen


def _alternate(centers, sides, time_budget, beta=800.0, max_iter=30):
    """Alternate optimizing centers and sides."""
    t0 = time.time()
    cur_c, cur_s = centers.copy(), sides.copy()
    cur_pen = _overlap_penalty_vec(cur_c[:, 0], cur_c[:, 1], cur_s, beta)
    best_c, best_s, best_pen = cur_c.copy(), cur_s.copy(), cur_pen

    for it in range(max_iter):
        if time.time() - t0 > time_budget * 0.85:
            break
        # optimize centers
        z0 = np.concatenate([cur_c[:, 0], cur_c[:, 1]])
        nc, pen_c = _optimize_centers(cur_s, z0, time_budget * 0.05, beta)
        if nc is not None:
            cur_c = nc
        # optimize sides
        ns, pen_s = _optimize_sides(cur_c, cur_s, time_budget * 0.05, beta)
        if ns is not None:
            cur_s = ns
        cur_pen = _overlap_penalty_vec(cur_c[:, 0], cur_c[:, 1], cur_s, beta)
        if cur_pen < best_pen - 1e-12:
            best_c, best_s, best_pen = cur_c.copy(), cur_s.copy(), cur_pen
        elif cur_pen >= best_pen:
            # no improvement, maybe stop
            if it > 3:
                break

    return best_c, best_s, best_pen


def squares_in_square10() -> tuple:
    n = 10
    deadline = time.time() + 620.0  # safety margin before 630s

    # Start from grid packing
    centers0, sides0 = _grid_packing()
    best_centers, best_sides = centers0.copy(), sides0.copy()
    best_score = float(np.sum(sides0))

    def _try_update(c, s):
        nonlocal best_centers, best_sides, best_score
        if c is None or s is None:
            return
        if not _is_valid(c, s):
            return
        sc = float(np.sum(s))
        if sc > best_score + 1e-12:
            best_centers, best_sides, best_score = c.copy(), s.copy(), sc

    # Phase 1: alternate optimization from grid start
    t0 = time.time()
    c, s, pen = _alternate(centers0, sides0, 120.0, beta=800.0, max_iter=40)
    _try_update(c, s)

    # Phase 2: multi-start with different initial layouts
    rng = np.random.default_rng(42)
    t1 = time.time()
    while time.time() < deadline - 30:
        # Generate a random-ish starting layout: jittered grid or random
        mode = rng.integers(0, 3)
        if mode == 0:
            # jittered grid
            cc, ss = _grid_packing()
            cc = cc + rng.normal(0, 0.03, cc.shape)
            cc = np.clip(cc, 0.05, 0.95)
            ss = ss + rng.normal(0, 0.02, len(ss))
            ss = np.clip(ss, 0.1, 0.5)
        elif mode == 1:
            # random positions, equal sides
            cc = rng.uniform(0.1, 0.9, (n, 2))
            ss = np.full(n, 0.25)
        else:
            # random positions, random sides
            cc = rng.uniform(0.1, 0.9, (n, 2))
            ss = rng.uniform(0.15, 0.4, n)

        if not _is_valid(cc, ss):
            # nudge to make valid: shrink sides
            ss = ss * 0.5
            if not _is_valid(cc, ss):
                cc, ss = _grid_packing()

        c2, s2, pen2 = _alternate(cc, ss, 30.0, beta=800.0, max_iter=20)
        _try_update(c2, s2)

        # Also try pure center optimization keeping sides from grid
        if time.time() < deadline - 30:
            z0 = np.concatenate([cc[:, 0], cc[:, 1]])
            nc, pc = _optimize_centers(ss, z0, 10.0, beta=800.0)
            if nc is not None and _is_valid(nc, ss):
                sc = float(np.sum(ss))
                if sc > best_score + 1e-12:
                    best_centers, best_sides, best_score = nc.copy(), ss.copy(), sc

    # Final polish: high-precision local solve from best
    if time.time() < deadline - 10:
        t2 = time.time()
        c3, s3, pen3 = _alternate(best_centers, best_sides, 20.0, beta=2000.0, max_iter=30)
        _try_update(c3, s3)

    # Ensure final result is valid; fallback to grid if not
    if not _is_valid(best_centers, best_sides):
        best_centers, best_sides = _grid_packing()

    return best_centers.astype(np.float64), best_sides.astype(np.float64)


# EVOLVE-BLOCK-END
