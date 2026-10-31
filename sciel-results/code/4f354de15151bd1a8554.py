"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13
DEADLINE = 142.0  # safety margin below the 150s wall-clock kill


def _score(pts):
    """min triangle area / convex hull area (scale-invariant). 0 if degenerate."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape[0] < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # 2D volume == area
    except Exception:
        return 0.0
    if hull_area <= 1e-12:
        return 0.0
    n = pts.shape[0]
    a = pts[:, 0]
    b = pts[:, 1]
    d = np.subtract.outer(a, a)  # d[i,j] = a[i]-a[j]
    e = np.subtract.outer(b, b)  # e[i,j] = b[i]-b[j]
    cross = d * e - e * d        # cross = (p_i - p_j) x (p_k - p_i)
    np.fill_diagonal(cross, 0.0)
    # |cross| = 2 * area of triangle (i, j, k)
    vals = np.abs(cross)
    kth = 2 * (n - 1) - 1        # 24th smallest of the 2*(n-1) values per row
    row_min = np.partition(vals, kth, axis=1)[:, kth]
    min_twice = row_min.min()
    min_area = 0.5 * min_twice
    return min_area / hull_area


def _objective(x):
    return -_score(x.reshape(-1, 2))


def _refine(pts):
    """Local refinement: Nelder-Mead then coordinate descent. Returns best points."""
    pts = np.asarray(pts, dtype=float).copy()
    best = pts
    bs = _score(pts)
    try:
        res = minimize(_objective, pts.ravel(), method='Nelder-Mead',
                       options={'maxiter': 600, 'xatol': 1e-6, 'fatol': 1e-9,
                                'adaptive': True})
        if res.fun < -bs + 1e-12:
            best = res.x.reshape(-1, 2)
            bs = -res.fun
    except Exception:
        pass
    # Coordinate descent on individual coordinates
    for it in range(120):
        improved = False
        for i in range(N):
            for c in range(2):
                base = best[i, c]
                for step in (0.05, 0.01, 0.002):
                    for s in (step, -step):
                        cand = best.copy()
                        cand[i, c] = base + s
                        s2 = _score(cand)
                        if s2 > bs + 1e-12:
                            best = cand
                            bs = s2
                            improved = True
        if not improved:
            break
    return best


def _ring(m, r, rot):
    ang = np.arange(m) * (2.0 * np.pi / m) + rot
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def _make_seeds():
    seeds = []
    # Concentric regular polygons
    for m1, m2 in [(6, 7), (5, 8), (7, 6), (4, 9), (9, 4), (3, 10)]:
        for r2 in (0.3, 0.5, 0.65, 0.8):
            for rot in (0.0, np.pi / m1, np.pi / m2):
                s = np.vstack([_ring(m1, 1.0, 0.0), _ring(m2, r2, rot)])
                seeds.append(s)
    # Points on a circle
    seeds.append(_ring(13, 1.0, 0.0))
    # Random seeds
    rng = np.random.default_rng(0)
    for _ in range(12):
        seeds.append(rng.random((N, 2)) * 2.0 - 1.0)
    return seeds


def construct():
    start = time.time()
    seeds = _make_seeds()
    best = None
    best_score = -1.0
    order = sorted(range(len(seeds)), key=lambda i: -_score(seeds[i]))
    for idx in order:
        if time.time() > DEADLINE:
            break
        cand = _refine(seeds[idx])
        s = _score(cand)
        if s > best_score:
            best_score = s
            best = cand
        if time.time() > DEADLINE:
            break
    if best is None:
        best = seeds[0]
    return np.asarray(best, dtype=float)
