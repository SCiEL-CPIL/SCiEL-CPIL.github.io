"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _all_triangle_areas(pts):
    """Vectorized areas of all C(n,3) triangles."""
    n = len(pts)
    parts = []
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            v2 = pts[j + 1:] - pts[i]
            if len(v2) == 0:
                continue
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            parts.append(0.5 * cross)
    return np.concatenate(parts)


def _convex_hull_area(pts):
    """Fast O(n^2) convex hull area via monotone chain + shoelace."""
    n = len(pts)
    if n < 3:
        return 0.0
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    p = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for i in range(n):
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p[i]) <= 0:
            lower.pop()
        lower.append(p[i])
    upper = []
    for i in range(n - 1, -1, -1):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p[i]) <= 0:
            upper.pop()
        upper.append(p[i])
    hull = lower[:-1] + upper[:-1]
    m = len(hull)
    if m < 3:
        return 0.0
    area = 0.0
    for i in range(m):
        j = (i + 1) % m
        area += hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1]
    return 0.5 * abs(area)


def _score(pts):
    """min triangle area / convex hull area (translation-invariant)."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2):
        return 0.0
    pts = pts - pts.mean(axis=0)  # centering: no-op for score, cleaner landscape
    areas = _all_triangle_areas(pts)
    min_area = float(np.min(areas))
    if min_area <= 0:
        return 0.0
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    return min_area / hull_area


def _neg_obj(params):
    pts = params.reshape(N, 2)
    s = _score(pts)
    if not np.isfinite(s):
        return 1e6
    # tiny regularization breaks translation symmetry (negligible on score)
    c = pts.mean(axis=0)
    return -s + 1e-8 * (c[0] * c[0] + c[1] * c[1])


def _init_1_plus_12():
    pts = np.zeros((N, 2))
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts[1:] = np.column_stack([np.cos(ang), np.sin(ang)])
    return pts


def _init_1_plus_6_6(ri, ro):
    pts = np.zeros((N, 2))
    ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ang_out = ang + np.pi / 6
    pts[1:7] = np.column_stack([ri * np.cos(ang), ri * np.sin(ang)])
    pts[7:13] = np.column_stack([ro * np.cos(ang_out), ro * np.sin(ang_out)])
    return pts


def _init_regular_13():
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _init_random_disk(rng):
    ang = rng.rand(N) * 2 * np.pi
    r = np.sqrt(rng.rand(N))
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def construct():
    t0 = time.time()
    deadline = t0 + 112.0  # safety margin under the 120s per-eval limit
    rng = np.random.RandomState(2024)

    # Structured inits: the 1+12 / 1+6+6 rings sit in the high-score basin.
    base_inits = [
        _init_1_plus_12(),
        _init_1_plus_6_6(0.50, 1.00),
        _init_1_plus_6_6(0.60, 1.00),
        _init_1_plus_6_6(0.45, 0.95),
        _init_1_plus_6_6(0.55, 1.05),
        _init_regular_13(),
    ]
    for _ in range(4):
        base_inits.append(_init_random_disk(rng))

    # Build start list: bases + small perturbations + pure random.
    starts = []
    for b in base_inits:
        starts.append(b.copy())
        for _ in range(2):
            starts.append(b + 0.06 * rng.randn(N, 2))
    for _ in range(3):
        starts.append(rng.rand(N, 2) * 2.0 - 1.0)

    best_score = -1.0
    best_pts = None

    # Multi-start Nelder-Mead (derivative-free -> exact non-smooth score is fine).
    for sp in starts:
        if time.time() > deadline - 10.0:
            break
        params = sp.reshape(-1)
        try:
            res = minimize(_neg_obj, params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10,
                                    'fatol': 1e-12, 'adaptive': True})
            cand = res.x.reshape(N, 2)
            sc = _score(cand)
            if sc > best_score:
                best_score = sc
                best_pts = cand.copy()
        except Exception:
            pass

    # Tight polish on the global best.
    if best_pts is not None and time.time() < deadline - 6.0:
        params = best_pts.reshape(-1)
        try:
            res = minimize(_neg_obj, params, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-12,
                                    'fatol': 1e-14, 'adaptive': True})
            cand = res.x.reshape(N, 2)
            sc = _score(cand)
            if sc > best_score:
                best_score = sc
                best_pts = cand.copy()
        except Exception:
            pass

        # Powell polish.
        if time.time() < deadline - 5.0:
            try:
                res = minimize(_neg_obj, params, method='Powell',
                               options={'maxiter': 300, 'xatol': 1e-12,
                                        'fatol': 1e-14})
                cand = res.x.reshape(N, 2)
                sc = _score(cand)
                if sc > best_score:
                    best_score = sc
                    best_pts = cand.copy()
            except Exception:
                pass

        # Jitter-restarts to escape shallow local minima near the best.
        for _ in range(3):
            if time.time() > deadline - 4.0:
                break
            jittered = best_pts + 0.03 * rng.randn(N, 2)
            try:
                res = minimize(_neg_obj, jittered.reshape(-1), method='Nelder-Mead',
                               options={'maxiter': 600, 'xatol': 1e-12,
                                        'fatol': 1e-14, 'adaptive': True})
                cand = res.x.reshape(N, 2)
                sc = _score(cand)
                if sc > best_score:
                    best_score = sc
                    best_pts = cand.copy()
            except Exception:
                pass

    # Fallback: guarantee a valid (N,2) point set with positive hull area.
    if best_pts is None or _convex_hull_area(best_pts) <= 1e-9:
        best_pts = _init_1_plus_12()

    return best_pts
# EVOLVE-BLOCK-END
