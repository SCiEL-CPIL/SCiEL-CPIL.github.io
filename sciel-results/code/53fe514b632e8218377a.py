"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = 140.0  # safety margin under 150s wall-clock


def _rect_dims(w, h):
    # perimeter = 2(w+h) = 4  =>  w + h = 2
    return w, h


def _validity(x, w, h):
    n = len(x)
    if n != N:
        return False
    for i in range(n):
        cx, cy, r = x[i]
        if r <= 0:
            return False
        if cx - r < -1e-9 or cx + r > w + 1e-9:
            return False
        if cy - r < -1e-9 or cy + r > h + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d2 = (x[i, 0] - x[j, 0]) ** 2 + (x[i, 1] - x[j, 1]) ** 2
            if d2 < (x[i, 2] + x[j, 2]) ** 2 - 1e-9:
                return False
    return True


def _score(x, w, h):
    if not _validity(x, w, h):
        return -np.inf
    return float(np.sum(x[:, 2]))


def _penalty_obj(x, w, h, lam=50.0):
    # objective to minimize = -sum(r) + lam * (violation terms)
    n = len(x)
    base = -np.sum(x[:, 2])
    pen = 0.0
    for i in range(n):
        cx, cy, r = x[i]
        # walls
        pen += max(0.0, r - cx) ** 2
        pen += max(0.0, r - (w - cx)) ** 2
        pen += max(0.0, r - cy) ** 2
        pen += max(0.0, r - (h - cy)) ** 2
    for i in range(n):
        for j in range(i + 1, n):
            d2 = (x[i, 0] - x[j, 0]) ** 2 + (x[i, 1] - x[j, 1]) ** 2
            d = np.sqrt(max(d2, 1e-12))
            pen += max(0.0, (x[i, 2] + x[j, 2]) - d) ** 2
    return base + lam * pen


def _make_grid(n, w, h):
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) * w / side
            cy = (j + 0.5) * h / side
            r = 0.5 * min(w, h) / side * 0.9
            cells.append([cx, cy, r])
    return np.array(cells[:n])


def _refit_and_grow(x, w, h, max_iter=400, dt=0.005):
    # Simple gradient ascent on radii with overlap/wall resolution.
    n = len(x)
    x = x.copy()
    for it in range(max_iter):
        # compute forces: each circle wants to grow; overlaps push apart/shrink
        grow = np.zeros(n)
        # wall constraint: max radius allowed by walls
        wall_r = np.zeros(n)
        for i in range(n):
            wall_r[i] = min(x[i, 0], w - x[i, 0], x[i, 1], h - x[i, 1])
        # overlap: for each pair, if overlapping, reduce both radii / push apart
        overlap = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i, 0] - x[j, 0]
                dy = x[i, 1] - x[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                need = x[i, 2] + x[j, 2]
                if d < need:
                    ov = need - d
                    overlap[i] += ov
                    overlap[j] += ov
        # update radii
        for i in range(n):
            # target radius limited by wall
            target = wall_r[i]
            # reduce due to overlap
            new_r = x[i, 2] + dt * (target - x[i, 2]) - dt * 0.5 * overlap[i]
            new_r = max(1e-4, min(new_r, wall_r[i]))
            x[i, 2] = new_r
    return x


def _solve_aspect(w, h, t0):
    x = _make_grid(N, w, h)
    x = _refit_and_grow(x, w, h, max_iter=600, dt=0.004)
    # local polish with scipy on radii only (positions fixed) to squeeze
    def obj(radii):
        xx = x.copy()
        xx[:, 2] = radii
        return _penalty_obj(xx, w, h, lam=80.0)
    r0 = x[:, 2].copy()
    try:
        res = minimize(obj, r0, method='L-BFGS-B',
                       bounds=[(1e-4, min(w, h) / 2)] * N,
                       options={'maxiter': 200, 'maxfun': 400})
        if res.x is not None:
            xx = x.copy()
            xx[:, 2] = np.clip(res.x, 1e-4, None)
            # ensure validity; if invalid, fall back to previous
            if _validity(xx, w, h):
                x = xx
    except Exception:
        pass
    return x


def construct():
    t0 = time.time()
    best = None
    best_score = -np.inf

    # Try a range of aspect ratios
    ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45]
    for ar in ratios:
        if time.time() - t0 > DEADLINE:
            break
        w = ar / (1 + ar) * 2
        h = 1 / (1 + ar) * 2
        # also swap orientation
        for (ww, hh) in [(w, h), (h, w)]:
            if time.time() - t0 > DEADLINE:
                break
            try:
                x = _solve_aspect(ww, hh, t0)
            except Exception:
                continue
            s = _score(x, ww, hh)
            if np.isfinite(s) and s > best_score:
                best_score = s
                best = (x.copy(), ww, hh)

    if best is None:
        # fallback to seed-like grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, w, h = best
    # Final safety: if somehow invalid, shrink radii slightly
    if not _validity(x, w, h):
        for i in range(N):
            x[i, 2] *= 0.999
        if not _validity(x, w, h):
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                     for i in range(side) for j in range(side)]
            return np.array(cells[:N])
    return x
