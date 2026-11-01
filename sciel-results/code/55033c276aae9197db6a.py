"""Circle packing in perimeter-4 rectangle: maximize sum of 21 radii."""
import numpy as np
import time

N = 21
DEADLINE = 115.0  # safety margin before 150s kill
_t0 = time.time()


def _elapsed():
    return time.time() - _t0


def _check_valid(p):
    """p: (N,3) [x,y,r]. Return True if disjoint + contained in [0,1]^2."""
    if p.shape != (N, 3):
        return False
    x, y, r = p[:, 0], p[:, 1], p[:, 2]
    if np.any(r <= 0):
        return False
    if np.any(x < -1e-9) or np.any(y < -1e-9) or np.any(x > 1 + 1e-9) or np.any(y > 1 + 1e-9):
        return False
    if np.any(x < r - 1e-9) or np.any(y < r - 1e-9) or np.any(1 - x < r - 1e-9) or np.any(1 - y < r - 1e-9):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    rr = r[:, None] + r[None, :]
    np.fill_diagonal(rr, 0.0)
    np.fill_diagonal(d, np.inf)
    if np.any(d + 1e-9 < rr):
        return False
    return True


def _objective_sum(v):
    """v: (3N,). Return -sum(r) for minimization."""
    r = v[2::2]
    return -np.sum(r)


def _constraints(v):
    """Inequality constraints (SLSQP requires g>=0):
    containment: x-r, 1-x-r, y-r, 1-y-r >= 0
    non-overlap: d_ij - (r_i+r_j) >= 0
    """
    x = v[0::2]
    y = v[1::2]
    r = v[2::2]
    cons = [x - r, 1.0 - x - r, y - r, 1.0 - y - r]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    rr = r[:, None] + r[None, :]
    np.fill_diagonal(d, np.inf)
    np.fill_diagonal(rr, 0.0)
    cons.append(d - rr)
    return np.concatenate([c.ravel() for c in cons])


def _solve_slsqp(p0):
    """SLSQP from p0 (N,3). Returns (N,3) valid or None."""
    from scipy.optimize import minimize
    v0 = p0.reshape(-1)
    try:
        res = minimize(_objective_sum, v0, method='SLSQP',
                       constraints=[{'type': 'ineq', 'fun': _constraints}],
                       bounds=[(0, 1)] * (2 * N) + [(1e-4, 0.5)] * N,
                       options={'maxiter': 400, 'ftol': 1e-10})
        if res.success or res.status in (0, 8):
            p = res.x.reshape(N, 3)
            if _check_valid(p):
                return p
    except Exception:
        pass
    return None


def _softmin_polish(p):
    """Smooth log-sum-exp (soft-min) polish on the nonsmooth min-margin objective."""
    from scipy.optimize import minimize

    def softmin_obj(v):
        xv = v[0::2]
        yv = v[1::2]
        rv = v[2::2]
        cm = np.concatenate([xv - rv, 1 - xv - rv, yv - rv, 1 - yv - rv])
        dx = xv[:, None] - xv[None, :]
        dy = yv[:, None] - yv[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        rr = rv[:, None] + rv[None, :]
        np.fill_diagonal(d, np.inf)
        np.fill_diagonal(rr, 0.0)
        gaps = (d - rr).ravel()
        margins = np.concatenate([cm, gaps])
        t = 0.05
        m = -margins / t
        m = m - m.max()
        lse = np.log(np.sum(np.exp(m))) + m.max()
        return -t * lse

    v0 = p.reshape(-1)
    try:
        res = minimize(softmin_obj, v0, method='SLSQP',
                       constraints=[{'type': 'ineq', 'fun': _constraints}],
                       bounds=[(0, 1)] * (2 * N) + [(1e-4, 0.5)] * N,
                       options={'maxiter': 200, 'ftol': 1e-12})
        if res.success or res.status in (0, 8):
            p2 = res.x.reshape(N, 3)
            if _check_valid(p2):
                return p2
    except Exception:
        pass
    return p


def _scale_radius(p):
    """Binary-search global scale factor (<=1) on radii to reach constraint boundary."""
    x, y = p[:, 0].copy(), p[:, 1].copy()
    r = p[:, 2].copy()
    lo, hi = 0.5, 1.0
    best_s = 0.5
    for _ in range(40):
        mid = (lo + hi) / 2
        rs = r * mid
        if np.any(x < rs - 1e-9) or np.any(y < rs - 1e-9) or np.any(1 - x < rs - 1e-9) or np.any(1 - y < rs - 1e-9):
            hi = mid
            continue
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        rr = rs[:, None] + rs[None, :]
        np.fill_diagonal(d, np.inf)
        np.fill_diagonal(rr, 0.0)
        if np.any(d + 1e-9 < rr):
            hi = mid
            continue
        best_s = mid
        lo = mid
    return np.column_stack([x, y, r * best_s])


def _initial_layouts(rng):
    """Diverse initial (N,3) layouts on [0,1]^2."""
    layouts = []
    side = 5
    r0 = 0.5 / side - 1e-3
    cells = [((i + 0.5) / side, (j + 0.5) / side, r0) for i in range(side) for j in range(side)]
    layouts.append(np.array(cells[:N]))
    jg = layouts[0].copy()
    jg[:, 0] += rng.uniform(-0.03, 0.03, N)
    jg[:, 1] += rng.uniform(-0.03, 0.03, N)
    jg[:, 0] = np.clip(jg[:, 0], 0.05, 0.95)
    jg[:, 1] = np.clip(jg[:, 1], 0.05, 0.95)
    layouts.append(jg)
    ring_r = 0.35
    ring_n = 16
    th = np.linspace(0, 2 * np.pi, ring_n, endpoint=False)
    rx = 0.5 + ring_r * np.cos(th)
    ry = 0.5 + ring_r * np.sin(th)
    rr = np.full(ring_n, 0.11)
    ring = np.column_stack([rx, ry, rr])
    center = np.array([[0.5, 0.5, 0.15]])
    layouts.append(np.vstack([center, ring])[:N])
    rx = rng.uniform(0.1, 0.9, N)
    ry = rng.uniform(0.1, 0.9, N)
    rr = np.full(N, 0.1)
    layouts.append(np.column_stack([rx, ry, rr]))
    return layouts


def construct():
    global _t0
    _t0 = time.time()
    rng = np.random.default_rng(12345)
    best = None
    best_val = -np.inf

    layouts = _initial_layouts(rng)
    starts = list(layouts)
    for base in layouts:
        for _ in range(2):
            pert = base.copy()
            pert[:, 0] += rng.uniform(-0.05, 0.05, N)
            pert[:, 1] += rng.uniform(-0.05, 0.05, N)
            pert[:, 0] = np.clip(pert[:, 0], 0.05, 0.95)
            pert[:, 1] = np.clip(pert[:, 1], 0.05, 0.95)
            starts.append(pert)

    for s in starts:
        if _elapsed() > DEADLINE - 5:
            break
        p = _solve_slsqp(s)
        if p is None:
            continue
        p2 = _softmin_polish(p)
        p3 = _scale_radius(p2)
        val = np.sum(p3[:, 2])
        if val > best_val:
            best_val = val
            best = p3
        val0 = np.sum(p[:, 2])
        if val0 > best_val:
            best_val = val0
            best = p

    if best is None or not _check_valid(best):
        side = 5
        r0 = 0.5 / side - 1e-3
        cells = [((i + 0.5) / side, (j + 0.5) / side, r0) for i in range(side) for j in range(side)]
        best = np.array(cells[:N])
        for s in np.linspace(1.0, 0.5, 20):
            p = best.copy()
            p[:, 2] *= s
            if _check_valid(p):
                best = p
                break

    return best
