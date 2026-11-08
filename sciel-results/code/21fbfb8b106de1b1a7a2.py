"""Circle packing in the unit square: maximize sum of radii for 26 disjoint circles."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
DEADLINE = 135.0  # safety margin well under the 150s kill
T0 = time.time()


def _time_left():
    return DEADLINE - (time.time() - T0)


def _valid(p):
    """Return True if packing p (n,3) is feasible (disjoint + inside unit square)."""
    if p.shape[0] != N:
        return False
    x = p[:, 0]; y = p[:, 1]; r = p[:, 2]
    if np.any(r <= 0.0):
        return False
    if np.any(x < -1e-6) or np.any(y < -1e-6):
        return False
    if np.any(x + r > 1.0 + 1e-6) or np.any(y + r > 1.0 + 1e-6):
        return False
    n = N
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]; dy = y[i] - y[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-7:
                return False
    return True


def _score(p):
    return float(np.sum(p[:, 2]))


def _hex_layout(rng):
    """Staggered hexagonal packing, rows of equal radius, sized to fit N circles."""
    rows = 5
    counts = []
    rem = N
    for k in range(rows):
        c = N // rows + (1 if k < N % rows else 0)
        counts.append(c)
        rem -= c
    while rem > 0:  # safety
        counts[-1] += 1
        rem -= 1

    best = None
    for r in np.linspace(0.03, 0.16, 900):
        dy = 2.0 * r
        dx = 1.7320508075688772 * r
        ok = True
        for m, c in enumerate(counts):
            if c == 0:
                continue
            y = r + m * dy
            if y + r > 1.0 + 1e-9:
                ok = False
                break
            x0 = r + (m % 2) * 0.5 * dx
            if x0 + (c - 1) * dx + r > 1.0 + 1e-9:
                ok = False
                break
        if not ok:
            continue
        if best is None or r > best[0]:
            best = (r, list(counts))
    if best is None:
        return None
    r, counts = best
    dy = 2.0 * r
    dx = 1.7320508075688772 * r
    pts = []
    for m, c in enumerate(counts):
        y = r + m * dy
        x0 = r + (m % 2) * 0.5 * dx
        for i in range(c):
            pts.append((x0 + i * dx, y, r))
    return np.array(pts[:N])


def _grid_layout():
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])


def _objective(v, W):
    p = v.reshape(-1, 3)
    x = p[:, 0]; y = p[:, 1]; r = p[:, 2]
    n = p.shape[0]
    pen = 0.0
    # boundary: x,y in [r, 1-r]
    pen += W * np.sum(np.maximum(0.0, r - x))
    pen += W * np.sum(np.maximum(0.0, r - y))
    pen += W * np.sum(np.maximum(0.0, x + r - 1.0))
    pen += W * np.sum(np.maximum(0.0, y + r - 1.0))
    # pairwise non-overlap
    for i in range(n):
        dx = x[i] - x[i + 1:]
        dy = y[i] - y[i + 1:]
        d = np.sqrt(dx * dx + dy * dy)
        pen += W * np.sum(np.maximum(0.0, r[i] + r[i + 1:] - d))
    return -np.sum(r) + pen


def _optimize(p0, rng, W):
    v0 = p0.reshape(-1)
    lb = np.full(3 * N, 1e-5)
    ub = np.array([1.0 - 1e-5, 1.0 - 1e-5, 0.5 - 1e-5] * N, dtype=float)
    res = minimize(_objective, v0, args=(W,), method='L-BFGS-B',
                   bounds=list(zip(lb, ub)), options={'maxiter': 300, 'ftol': 1e-9})
    p = res.x.reshape(-1, 3)
    return p


def construct():
    global T0
    T0 = time.time()
    rng = np.random.default_rng(12345)

    hexl = _hex_layout(rng)
    base = hexl if hexl is not None else _grid_layout()

    W = 50.0
    best_p = base.copy()
    best_s = _score(best_p)

    # polish the base layout directly
    if _time_left() > 6.0:
        cand = _optimize(base, rng, W)
        if _valid(cand):
            s = _score(cand)
            if s > best_s:
                best_s, best_p = s, cand.copy()

    # many-restart local search
    while _time_left() > 15.0:
        if rng.random() < 0.85:
            p0 = base + rng.normal(0.0, 0.02, size=base.shape)
        else:
            p0 = np.column_stack([
                rng.uniform(0.05, 0.95, N),
                rng.uniform(0.05, 0.95, N),
                rng.uniform(0.03, 0.12, N),
            ])
        cand = _optimize(p0, rng, W)
        if _valid(cand):
            s = _score(cand)
            if s > best_s:
                best_s, best_p = s, cand.copy()

    # final safety: ensure we return something valid
    if not _valid(best_p):
        g = _grid_layout()
        if _valid(g):
            best_p = g
        else:
            # guaranteed-valid tiny circles
            best_p = np.column_stack([
                np.linspace(0.05, 0.95, N),
                np.linspace(0.95, 0.05, N),
                np.full(N, 0.015),
            ])
    return best_p
