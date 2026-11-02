"""Structural redesign: variable-radius circle packing via SLSQP with
multi-stage hexagonal initialization and a deadline guard."""
import time
import numpy as np

N = 26
DEADLINE = 150.0
SAFETY = 12.0  # seconds of margin before the 150 s kill
TIME_LIMIT = DEADLINE - SAFETY


def _feasible(x, y, r, eps=1e-7):
    """Return True if all circles are inside [0,1]^2 and disjoint."""
    if np.any(r < -eps) or np.any(x < -eps) or np.any(x > 1 + eps) \
            or np.any(y < -eps) or np.any(y > 1 + eps):
        return False
    r = np.clip(r, 0.0, None)
    n = len(r)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - eps:
                return False
    return True


def _score(x, y, r):
    return float(np.sum(np.clip(r, 0.0, None)))


def _hex_init():
    """Hexagonal (close-packed) lattice init: rows of circles, variable radii."""
    r0 = 0.10
    rows, cols = 5, 5
    xs, ys = [], []
    dx = 2.0 * r0
    dy = np.sqrt(3.0) * r0
    for row in range(rows):
        y = 0.5 + row * dy
        for col in range(cols):
            x = 0.5 + col * dx + (row % 2) * r0
            xs.append(x)
            ys.append(y)
    # take the N closest to center
    idx = np.argsort([np.hypot(a - 0.5, b - 0.5) for a, b in zip(xs, ys)][:N])
    return np.array([xs[i] for i in idx]), np.array([ys[i] for i in idx]), \
        np.full(N, r0)


def _grid_init():
    """Grid init as a fallback / diversity seed."""
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    cells = cells[:N]
    return (np.array([c[0] for c in cells]), np.array([c[1] for c in cells]),
            np.array([c[2] for c in cells]))


def _objective(v):
    x, y, r = v[:N], v[N:2 * N], v[2 * N:3 * N]
    # squared overlap penalty (smooth, differentiable)
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            gap = np.hypot(x[i] - x[j], y[i] - y[j]) - r[i] - r[j]
            if gap < 0:
                pen += gap * gap
    # keep circles inside the square (soft penalty)
    for i in range(N):
        for b in (x[i] - r[i], 1.0 - (x[i] + r[i]),
                  y[i] - r[i], 1.0 - (y[i] + r[i])):
            if b < 0:
                pen += b * b
    return -np.sum(r) + 10.0 * pen


def _bounds():
    b = []
    for i in range(N):
        b.append((0.0, 1.0))   # x
    for i in range(N):
        b.append((0.0, 1.0))   # y
    for i in range(N):
        b.append((1e-6, 0.5))  # r
    return b


def _slsqp(x0, y0, r0, time_left):
    """Run SLSQP from a start point; return (x,y,r) or None."""
    from scipy.optimize import minimize
    v0 = np.concatenate([x0, y0, r0])
    t0 = time.time()
    try:
        res = minimize(_objective, v0, method='SLSQP',
                       bounds=_bounds(),
                       options={'maxiter': 300, 'ftol': 1e-9,
                                'disp': False,
                                'eps': 1e-4,
                                'time_limit': max(1.0, time_left - 2.0)})
        v = res.x
        x, y, r = v[:N].copy(), v[N:2 * N].copy(), v[2 * N:3 * N].copy()
        # clamp radii to a feasible range and clip tiny negatives
        r = np.clip(r, 1e-6, 0.5)
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        return x, y, r
    except Exception:
        return None


def construct():
    start = time.time()
    best = None
    best_val = -1.0

    # ---- Stage 1: hexagonal init + local refinement ----
    hx, hy, hr = _hex_init()
    t_left = TIME_LIMIT - (time.time() - start)
    if t_left > 5.0:
        out = _slsqp(hx, hy, hr, t_left)
        if out is not None:
            x, y, r = out
            if _feasible(x, y, r):
                val = _score(x, y, r)
                if val > best_val:
                    best_val, best = val, (x.copy(), y.copy(), r.copy())

    # ---- Stage 2: grid init + refinement (diversity) ----
    t_left = TIME_LIMIT - (time.time() - start)
    if t_left > 5.0:
        gx, gy, gr = _grid_init()
        out = _slsqp(gx, gy, gr, t_left)
        if out is not None:
            x, y, r = out
            if _feasible(x, y, r):
                val = _score(x, y, r)
                if val > best_val:
                    best_val, best = val, (x.copy(), y.copy(), r.copy())

    # ---- Stage 3: global refinement from the best feasible point ----
    t_left = TIME_LIMIT - (time.time() - start)
    if best is not None and t_left > 5.0:
        x, y, r = best
        out = _slsqp(x, y, r, t_left)
        if out is not None:
            x2, y2, r2 = out
            if _feasible(x2, y2, r2):
                val = _score(x2, y2, r2)
                if val > best_val:
                    best_val, best = val, (x2.copy(), y2.copy(), r2.copy())

    # ---- Fallback: guarantee a valid result ----
    if best is None:
        x, y, r = _grid_init()
        # shrink slightly to be safe
        r = np.clip(r, 1e-6, 0.5)
        best = (x, y, r)

    x, y, r = best
    r = np.clip(r, 1e-6, 0.5)
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)

    rows = np.stack([x, y, r], axis=1)
    return rows
