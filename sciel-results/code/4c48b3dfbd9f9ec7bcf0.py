import numpy as np
from scipy.optimize import minimize
import time

N = 32
TIME_LIMIT = 142.0  # evaluator kills at 150s; keep a safety margin

def _elapsed(start):
    return time.time() - start

def _is_valid(x, y, r, eps=1e-7):
    if np.any(x < -eps) or np.any(x > 1.0 + eps):
        return False
    if np.any(y < -eps) or np.any(y > 1.0 + eps):
        return False
    if np.any(r < -eps):
        return False
    if np.any(x < r - eps) or np.any(x > 1.0 - r + eps):
        return False
    if np.any(y < r - eps) or np.any(y > 1.0 - r + eps):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < (r[i] + r[j]) - eps:
                return False
    return True

def _sum_r(r):
    return float(np.sum(r))

def _fast_valid(x, y, r):
    """Vectorized validity check (tolerance 1e-9)."""
    if np.any(r < -1e-9):
        return False
    if np.any(x < r - 1e-9) or np.any(x > 1.0 - r + 1e-9):
        return False
    if np.any(y < r - 1e-9) or np.any(y > 1.0 - r + 1e-9):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy + 1e-14)
    need = r[:, None] + r[None, :]
    return bool(np.all(dist >= need - 1e-9))

def _objective(params, start):
    if _elapsed(start) > TIME_LIMIT:
        return 1e10
    x = params[:N]
    y = params[N:2 * N]
    r = params[2 * N:3 * N]
    r = np.maximum(r, 0.0)
    obj = -np.sum(r)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy + 1e-14)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    smooth_max = 0.5 * (diff + np.sqrt(diff * diff + 1e-8))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])

    def sm(v):
        return 0.5 * (v + np.sqrt(v * v + 1e-8))
    bound_pen = (np.sum(sm(r - x)) + np.sum(sm(x - (1.0 - r))) +
                 np.sum(sm(r - y)) + np.sum(sm(y - (1.0 - r))))

    mu = 50.0
    return obj + mu * (overlap_pen + bound_pen)

def _objective_tight(params, start):
    if _elapsed(start) > TIME_LIMIT:
        return 1e10
    x = params[:N]
    y = params[N:2 * N]
    r = params[2 * N:3 * N]
    r = np.maximum(r, 0.0)
    obj = -np.sum(r)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy + 1e-14)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    smooth_max = 0.5 * (diff + np.sqrt(diff * diff + 1e-12))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])

    def sm(v):
        return 0.5 * (v + np.sqrt(v * v + 1e-12))
    bound_pen = (np.sum(sm(r - x)) + np.sum(sm(x - (1.0 - r))) +
                 np.sum(sm(r - y)) + np.sum(sm(y - (1.0 - r))))

    mu = 100.0
    return obj + mu * (overlap_pen + bound_pen)

def _pbd_pass(x, y, r, passes=40):
    """Vectorized Position-Based Dynamics: project pairwise non-overlap and
    boundary constraints until (locally) satisfied. Radii are never shrunk."""
    rsum = r[:, None] + r[None, :]
    for _ in range(passes):
        # pairwise non-overlap
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx * dx + dy * dy
        d = np.sqrt(d2 + 1e-14)
        gap = rsum - d
        mask = (gap > 1e-12) & (np.abs(np.arange(N)[:, None]) > 0)
        mask = (gap > 1e-12) & (np.triu(np.ones((N, N), dtype=bool), k=1))
        if np.any(mask):
            with np.errstate(divide='ignore', invalid='ignore'):
                inv = np.where(d > 1e-12, 1.0 / d, 0.0)
            ux = dx * inv
            uy = dy * inv
            # where centers coincide, use a deterministic direction
            deg = (d <= 1e-12) & mask
            if np.any(deg):
                ii, jj = np.where(deg)
                ang = (ii * 2.399 + jj * 1.618) % (2 * np.pi)
                ux[ii, jj] = np.cos(ang)
                uy[ii, jj] = np.sin(ang)
            wi = (r[None, :] / rsum) * mask      # weight for i
            wj = (r[:, None] / rsum) * mask      # weight for j
            corr = gap * mask
            disp_x = (wi * corr * ux).sum(axis=1) - (wj * corr * ux).sum(axis=0)
            disp_y = (wi * corr * uy).sum(axis=1) - (wj * corr * uy).sum(axis=0)
            x += disp_x
            y += disp_y
        # boundary constraints
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)

def _physics_optimize(x, y, r, start, max_iter=20000):
    """Grow radii while projecting to satisfy constraints (PBD)."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)

    best_score = _sum_r(r)
    best_state = (x.copy(), y.copy(), r.copy())

    growth = 0.0015
    for it in range(max_iter):
        if _elapsed(start) > TIME_LIMIT - 6:
            break

        r = r + growth
        # cap radii by distance to walls
        cap = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        r = np.minimum(r, cap)

        _pbd_pass(x, y, r, passes=30)

        if _fast_valid(x, y, r):
            s = _sum_r(r)
            if s > best_score + 1e-11:
                best_score = s
                best_state = (x.copy(), y.copy(), r.copy())

    return best_state

def _make_valid(x, y, r):
    """Guarantee validity: push apart, then shrink any residual overlap/bounds."""
    x = x.copy()
    y = y.copy()
    r = r.copy()

    for _ in range(30):
        _pbd_pass(x, y, r, passes=100)
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        if _fast_valid(x, y, r):
            break

    if not _fast_valid(x, y, r):
        for _ in range(800):
            for i in range(N):
                mr = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
                if r[i] > mr * 0.999:
                    r[i] = max(0.0, mr * 0.998)
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx * dx + dy * dy + 1e-14)
            need = r[:, None] + r[None, :]
            overlaps = dist < need - 1e-10
            if not np.any(overlaps):
                break
            idx = np.where(overlaps)
            i, j = idx[0][0], idx[1][0]
            if j < i:
                i, j = j, i
            r[i] *= 0.99
            r[j] *= 0.99

    return x, y, r

def _hex_init():
    pts = []
    rowh = np.sqrt(3.0) / 2.0
    row = 0
    while len(pts) < N:
        for c in range(0, 12):
            x = c + (0.5 if (row % 2) else 0.0)
            y = row * rowh
            pts.append([x, y])
            if len(pts) >= N:
                break
        row += 1
    pts = np.array(pts[:N])
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin)
    if span < 1e-9:
        span = 1.0
    scale = 0.85 / span
    x = (pts[:, 0] - xmin) * scale + 0.075
    y = (pts[:, 1] - ymin) * scale + 0.075
    r = np.zeros(N)
    for i in range(N):
        mn = 1e9
        for j in range(N):
            if i != j:
                mn = min(mn, np.hypot(x[i] - x[j], y[i] - y[j]))
        r[i] = 0.49 * mn
    r = np.minimum(r, np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)))
    return x, y, r

def _rect_init(cols, rows):
    cells = []
    for j in range(rows):
        for i in range(cols):
            if len(cells) < N:
                x = (i + 0.5) / cols
                y = (j + 0.5) / rows
                cells.append([x, y])
    x = np.array([c[0] for c in cells])
    y = np.array([c[1] for c in cells])
    r = np.full(N, 0.5 / max(cols, rows) * 0.98)
    r = np.minimum(r, np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)))
    return x, y, r

def _random_init(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.15, 0.85, N)
    y = rng.uniform(0.15, 0.85, N)
    r = np.full(N, 0.07)
    r = np.minimum(r, np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)))
    return x, y, r

def _optimize(x, y, r, start):
    best = (x.copy(), y.copy(), r.copy())
    best_score = _sum_r(r)

    # Physics-based optimization (fast, vectorized)
    if _elapsed(start) < TIME_LIMIT - 8:
        try:
            res = _physics_optimize(x, y, r, start, max_iter=20000)
            if _fast_valid(*res):
                s = _sum_r(res[2])
                if s > best_score:
                    best_score = s
                    best = res
        except Exception:
            pass

    # Nelder-Mead
    if _elapsed(start) < TIME_LIMIT - 4:
        try:
            params = np.concatenate([best[0], best[1], best[2]])
            res = minimize(_objective, params, args=(start,),
                           method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-7,
                                    'fatol': 1e-9, 'adaptive': True})
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2 * N], res.x[2 * N:3 * N])
            if _fast_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass

    # Powell polish
    if _elapsed(start) < TIME_LIMIT - 2:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(_objective, p0, args=(start,),
                           method='Powell',
                           options={'maxiter': 1500, 'ftol': 1e-10})
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2 * N], res.x[2 * N:3 * N])
            if _fast_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass

    return best

def _tight_polish(x, y, r, start):
    best = (x.copy(), y.copy(), r.copy())
    best_score = _sum_r(r)

    if _elapsed(start) < TIME_LIMIT - 3:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(_objective_tight, p0, args=(start,),
                           method='Powell',
                           options={'maxiter': 3000, 'ftol': 1e-14, 'xtol': 1e-14})
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2 * N], res.x[2 * N:3 * N])
            if _fast_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass

    if _elapsed(start) < TIME_LIMIT - 2:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(_objective_tight, p0, args=(start,),
                           method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12,
                                    'fatol': 1e-14, 'adaptive': True})
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2 * N], res.x[2 * N:3 * N])
            if _fast_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass

    if _elapsed(start) < TIME_LIMIT - 1:
        rng = np.random.RandomState(42)
        for trial in range(120):
            if _elapsed(start) > TIME_LIMIT - 1:
                break
            scale = 0.005 * (1.0 - trial / 120.0)
            px = best[0] + rng.normal(0, scale, N)
            py = best[1] + rng.normal(0, scale, N)
            pr = np.maximum(best[2] + rng.normal(0, scale * 0.3, N), 1e-4)
            vx, vy, vr = _make_valid(px, py, pr)
            if _fast_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)

    return best

def _final_polish(x, y, r, start):
    best = (x.copy(), y.copy(), r.copy())
    best_score = _sum_r(r)

    for ftol in [1e-10, 1e-12]:
        if _elapsed(start) > TIME_LIMIT - 8:
            break
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(_objective, p0, args=(start,),
                           method='Powell',
                           options={'maxiter': 2000, 'ftol': ftol})
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2 * N], res.x[2 * N:3 * N])
            if _fast_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass

    if _elapsed(start) < TIME_LIMIT - 5:
        try:
            res = _tight_polish(*best, start)
            if _fast_valid(*res):
                s = _sum_r(res[2])
                if s > best_score:
                    best_score = s
                    best = res
        except Exception:
            pass

    if _elapsed(start) < TIME_LIMIT - 2:
        rng = np.random.RandomState(123)
        for k in range(60):
            if _elapsed(start) > TIME_LIMIT - 2:
                break
            scale = 0.01 * (1 - k / 60.0)
            px = best[0] + rng.normal(0, scale, N)
            py = best[1] + rng.normal(0, scale, N)
            pr = np.maximum(best[2] + rng.normal(0, scale * 0.5, N), 1e-4)
            vx, vy, vr = _make_valid(px, py, pr)
            if _fast_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)

    return best

def construct():
    start = time.time()
    inits = [
        _hex_init(),
        _rect_init(6, 6),
        _rect_init(7, 5),
        _rect_init(5, 7),
        _rect_init(8, 4),
        _rect_init(4, 8),
        _rect_init(9, 4),
        _rect_init(4, 9),
        _random_init(1),
        _random_init(7),
    ]

    best = inits[0]
    if not _fast_valid(*best):
        best = _make_valid(*best)
    best_score = _sum_r(best[2])

    for k, (x, y, r) in enumerate(inits):
        if _elapsed(start) > TIME_LIMIT - 20:
            break
        res = _optimize(x, y, r, start)
        if _fast_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best_score = s
                best = res

    if _elapsed(start) < TIME_LIMIT - 6:
        res = _final_polish(*best, start)
        if _fast_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best_score = s
                best = res

    x, y, r = _make_valid(*best)
    return np.column_stack((x, y, r))
