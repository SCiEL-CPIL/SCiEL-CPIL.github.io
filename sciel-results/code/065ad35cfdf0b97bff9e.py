"""Heilbronn convex regions: 13 points to maximize minimum triangle area / hull area.

Structural approach: represent 13 points as vertices of a convex polygon defined by
13 support lines with fixed evenly-spaced outward normals; optimize the line offsets.
Use a smooth (soft-min) penalty objective and fast gradient-based L-BFGS-B with
annealed restarts.
"""
import time
import warnings
import numpy as np
from itertools import combinations

warnings.filterwarnings("ignore")

N = 13
T0 = time.time()
DEADLINE = T0 + 140.0  # safety margin before the 150s kill

# Precompute the C(13,3) triangle index triples
_TRI = np.array(list(combinations(range(N), 3)), dtype=np.int64)  # (286, 3)


def _triangle_areas_vectorized(pts):
    """Area of every triangle among the N points, vectorized. Returns (286,)."""
    a = pts[_TRI[:, 0]]
    b = pts[_TRI[:, 1]]
    c = pts[_TRI[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                        (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def _hull_area(pts):
    """Area of the convex hull (scipy)."""
    from scipy.spatial import ConvexHull
    try:
        h = ConvexHull(pts)
        return float(h.volume)
    except Exception:
        return 1e-10


def _score(pts):
    """Minimum triangle area divided by convex hull area."""
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    return float(np.min(_triangle_areas_vectorized(pts)) / ha)


def _neg_score(pts):
    return -_score(pts)


def _soft_min_value(vals, T):
    """Smooth approximation to min(vals)."""
    m = np.min(vals)
    return m - T * np.log(np.sum(np.exp((vals - m) / T)))


def _soft_score(pts, T):
    """Smoothed objective: soft-min triangle area / hull area."""
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    return _soft_min_value(_triangle_areas_vectorized(pts), T) / ha


def _neg_soft_score(pts, T):
    return -_soft_score(pts, T)


# ---- Convex polygon via support lines (offsets) with fixed evenly-spaced normals ----
# Outward unit normals at fixed angles; vertex i is intersection of lines i-1 and i.
_NORMALS = np.column_stack([np.cos(np.linspace(0, 2 * np.pi, N, endpoint=False)),
                            np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))])


def _polygon_from_offsets(offsets):
    """Vertices of the convex polygon defined by support lines n_i . x = offsets[i].

    Vertex i = intersection of lines (i-1) and i.
    """
    d = _NORMALS
    o = np.asarray(offsets)
    n = N
    i1 = np.arange(n - 1)          # line index i-1
    i2 = np.arange(1, n)           # line index i
    a = d[i1]                      # (n-1,2)
    b = d[i2]                      # (n-1,2)
    ao = o[i1]
    bo = o[i2]
    det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    x = (ao * b[:, 1] - bo * a[:, 1]) / det
    y = (a[:, 0] * bo - b[:, 0] * ao) / det
    V1 = np.column_stack([x, y])            # vertices 0..n-2
    # vertex n-1 = intersection of lines n-1 and 0
    a = d[n - 1]
    b = d[0]
    det = a[0] * b[1] - a[1] * b[0]
    det = 1e-12 if abs(det) < 1e-12 else det
    x = (o[n - 1] * b[1] - o[0] * a[1]) / det
    y = (a[0] * o[0] - b[0] * o[n - 1]) / det
    V2 = np.array([[x, y]])
    return np.vstack([V1, V2])


def _select_n_vertices(poly, n):
    """Pick n vertices from a convex polygon (in order), evenly spaced, preserving orientation."""
    n_poly = len(poly)
    if n_poly == n:
        return poly.copy()
    if n_poly < n:
        return poly
    # indices of n equally spaced vertices around the polygon
    idx = np.round(np.linspace(0, n_poly, n, endpoint=False)).astype(int) % n_poly
    # ensure strictly increasing (monotone) to keep orientation
    idx = np.sort(np.unique(idx))
    if len(idx) < n:
        # fallback: take first n
        idx = np.arange(n)
    return poly[idx]


def _score_from_offsets(offsets):
    poly = _polygon_from_offsets(offsets)
    pts = _select_n_vertices(poly, N)
    return _score(pts)


def _points_from_offsets(offsets):
    poly = _polygon_from_offsets(offsets)
    return _select_n_vertices(poly, N)


def _offsets_from_points(pts):
    """Extract support-line offsets for the N evenly-spaced normal directions."""
    pts = np.asarray(pts)
    return pts @ _NORMALS  # n_i . x_i for each vertex


def _seed_regular():
    """Regular 13-gon (unit circumradius) as initial offsets."""
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(ang), np.sin(ang)])
    return _offsets_from_points(pts)


def construct():
    rng = np.random.RandomState(12345)
    from scipy.optimize import minimize

    best_score = -np.inf
    best_pts = None

    def consider(pts):
        nonlocal best_score, best_pts
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # ---- Stage 1: gradient-based annealing on the offset representation ----
    # Start from the regular polygon offsets.
    base = _seed_regular()
    consider(_points_from_offsets(base))

    # Anneal the temperature of the soft-min and re-optimize offsets each stage.
    temps = [0.05, 0.02, 0.01, 0.005, 0.002]
    cur = base.copy()
    for T in temps:
        if time.time() > DEADLINE:
            break
        try:
            res = minimize(_neg_soft_score, cur, args=(T,), method='L-BFGS-B',
                           options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-6,
                                    'maxls': 40})
            cur = res.x
            pts = _points_from_offsets(cur)
            consider(pts)
        except Exception:
            pass

    # ---- Stage 2: local exact-score refinement (Nelder-Mead) on offsets ----
    if best_pts is not None and time.time() < DEADLINE - 8:
        start = _offsets_from_points(best_pts)
        try:
            res = minimize(_neg_score, start, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-12,
                                    'adaptive': True})
            consider(_points_from_offsets(res.x))
        except Exception:
            pass

    # ---- Stage 3: random restarts (annealed soft-min + local polish) ----
    n_restarts = 12
    for r in range(n_restarts):
        if time.time() > DEADLINE - 20:
            break
        # Perturbed seed: regular polygon with random angular jitter + scale
        ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
        jitter = rng.normal(0, 0.15, size=N)
        ang2 = ang + jitter
        scale = rng.uniform(0.7, 1.3)
        pts = scale * np.column_stack([np.cos(ang2), np.sin(ang2)])
        consider(pts)
        start = _offsets_from_points(_points_from_offsets(_offsets_from_points(pts)))
        # annealed soft-min
        for T in [0.03, 0.01, 0.003]:
            if time.time() > DEADLINE - 20:
                break
            try:
                res = minimize(_neg_soft_score, start, args=(T,), method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-6,
                                        'maxls': 40})
                start = res.x
                consider(_points_from_offsets(start))
            except Exception:
                break
        # local polish on exact score
        if time.time() > DEADLINE - 15:
            break
        try:
            res = minimize(_neg_score, start, method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11,
                                    'adaptive': True})
            consider(_points_from_offsets(res.x))
        except Exception:
            pass

    # ---- Final: ensure we return a valid set of N points ----
    if best_pts is None:
        best_pts = _points_from_offsets(_seed_regular())

    # Safety: guarantee N points, finite, and a nonzero hull area.
    best_pts = np.asarray(best_pts, dtype=float)
    if best_pts.shape != (N, 2) or not np.all(np.isfinite(best_pts)):
        ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(ang), np.sin(ang)])

    return best_pts


# EVOLVE-BLOCK-END
