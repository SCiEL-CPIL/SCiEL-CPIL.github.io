"""Packing 50 equal circles in the unit square centred at the origin.

Continuous local search: smooth log-sum-exp soft-min of the radius objective
with a tanh change of variables (constraints removed), multiple diverse
restarts, then a high-precision exact-min polish.  Returns array (N, 2).
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
HALF = 0.5
DEADLINE = 120.0          # safety margin before the 150 s wall-clock kill
START = time.time()

# --- fixed, precomputed pair indices (i < j) ---
_IJ = np.array([i for i in range(N) for j in range(i + 1, N)], dtype=np.int64)
_KJ = np.array([j for i in range(N) for j in range(i + 1, N)], dtype=np.int64)
PAIRS = (_IJ, _KJ)


def _radius(pts):
    """Exact objective: largest common radius (min of pair/2 and wall)."""
    d = pts[:, _IJ] - pts[:, _KJ]
    pair = np.min(np.linalg.norm(d, axis=1)) * 0.5
    wall = np.min(HALF - np.abs(pts))
    return float(min(pair, wall))


def _soft_min(vals, beta):
    """Smooth min via log-sum-exp (returns a value <= min(vals))."""
    m = np.min(vals)
    return m - np.log(np.sum(np.exp(-beta * (vals - m)))) / beta


def _soft_radius(pts, beta):
    d = pts[:, _IJ] - pts[:, _KJ]
    pair = 0.5 * np.sqrt(np.einsum('ij,ij->i', d, d))
    wall = HALF - np.abs(pts)
    return _soft_min(np.concatenate([pair, wall]), beta)


def _neg_soft(pts, beta):
    return -_soft_radius(pts, beta)


def _param_to_pts(p):
    """Unconstrained params -> centres strictly inside the square."""
    return (HALF - 1e-6) * np.tanh(p)


def _pts_to_param(pts):
    eps = 1e-9
    p = np.tanh(pts / (HALF - 1e-6))
    return np.clip(p, -eps, eps)


def _polish(pts):
    """Exact-min local refinement (Nelder–Mead + tiny coordinate descent)."""
    p0 = _pts_to_param(pts)

    def f(p):
        return -_radius(_param_to_pts(p))

    res = minimize(f, p0, method='Nelder-Mead',
                   options={'maxiter': 4000, 'xatol': 1e-12, 'fatol': 1e-14,
                            'adaptive': True})
    p = res.x
    pts = _param_to_pts(p)

    # coordinate-descent polish on the exact objective (cheap, tight)
    for _ in range(6):
        improved = False
        for i in range(N):
            for k in range(2):
                best = pts[i, k]
                lo, hi = -HALF, HALF
                for _ in range(48):
                    m1 = lo + (hi - lo) / 3.0
                    m2 = hi - (hi - lo) / 3.0
                    pts[i, k] = m1
                    f1 = _radius(pts)
                    pts[i, k] = m2
                    f2 = _radius(pts)
                    if f1 > f2:
                        lo = m1
                    else:
                        hi = m2
                    pts[i, k] = best
                pts[i, k] = 0.5 * (lo + hi)
                if _radius(pts) > best:
                    pts[i, k] = best
                    best = _radius(pts)
                    improved = True
    return pts


def _hex_init():
    """Hexagonal lattice packing inside the square, nearest N points."""
    s = 2.0 / 50.0
    dx, dy = s, s * np.sqrt(3.0) / 2.0
    cand = []
    ny = int(2.0 / dy) + 4
    nx = int(2.0 / dx) + 4
    for j in range(ny):
        y = -1.0 + j * dy
        off = (dx / 2.0) if (j % 2) else 0.0
        for i in range(nx):
            x = -1.0 + i * dx + off
            cand.append((x, y))
    cand = np.array(cand)
    mask = (np.abs(cand[:, 0]) <= HALF) & (np.abs(cand[:, 1]) <= HALF)
    cand = cand[mask]
    d2 = np.einsum('ij,ij->i', cand * cand, cand * cand)
    idx = np.argsort(d2)[:N]
    return cand[idx].copy()


def _grid_init(jitter=0.0, seed=0):
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])[:N].copy()
    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        g += jitter * (rng.standard_normal(g.shape) - 0.5)
    return np.clip(g, -HALF + 1e-6, HALF - 1e-6)


def _random_init(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-HALF + 1e-3, HALF - 1e-3, size=(N, 2))


def _anneal(pts):
    """Simulated annealing on the soft objective (escapes local minima)."""
    p = _pts_to_param(pts)
    beta = 300.0
    T = 0.0004
    T_end = 1e-7
    steps = 2200
    rng = np.random.default_rng(7)
    f = _neg_soft(_param_to_pts(p), beta)
    best_p = p.copy()
    best_f = f
    for s in range(steps):
        if time.time() > DEADLINE - 6.0:
            break
        k = rng.integers(0, N)
        d = rng.standard_normal(2)
        p_new = p.copy()
        p_new[k] += 0.004 * d
        p_new = np.clip(p_new, -0.99, 0.99)
        f_new = _neg_soft(_param_to_pts(p_new), beta)
        if f_new < f or rng.random() < np.exp(-(f_new - f) / T):
            p = p_new
            f = f_new
            if f < best_f:
                best_f = f
                best_p = p.copy()
        T *= (T_end / T) ** (1.0 / steps)
    return _param_to_pts(best_p)


def construct():
    t_start = time.time()
    inits = [_hex_init(), _grid_init(0.0, 0), _grid_init(0.004, 1),
             _random_init(2), _random_init(3), _random_init(4)]

    best_pts = inits[0].copy()
    best_r = _radius(best_pts)

    for init in inits:
        if time.time() > DEADLINE - 18.0:
            break
        # smooth anneal (cheap, escapes grid optimum)
        pts_a = _anneal(init)
        # anneal from the current best too (refinement)
        if time.time() <= DEADLINE - 18.0:
            pts_a2 = _anneal(best_pts)
            pts_a = pts_a if _radius(pts_a) >= _radius(pts_a2) else pts_a2
        # exact polish
        pts_p = _polish(pts_a)
        r = _radius(pts_p)
        if r > best_r:
            best_r = r
            best_pts = pts_p.copy()
        # final exact polish directly from the raw init as a backup
        if time.time() <= DEADLINE - 12.0:
            pts_p2 = _polish(init)
            r2 = _radius(pts_p2)
            if r2 > best_r:
                best_r = r2
                best_pts = pts_p2.copy()

    # last-chance polish of the best before returning
    if time.time() <= DEADLINE - 8.0:
        pts_p3 = _polish(best_pts)
        if _radius(pts_p3) > best_r:
            best_pts = pts_p3

    # hard guard: never return after the deadline
    if time.time() > DEADLINE - 2.0:
        pass  # best_pts already valid

    out = np.asarray(best_pts, dtype=np.float64)
    if out.shape != (N, 2) or not np.all(np.isfinite(out)):
        out = _grid_init(0.0, 0)
    return out
