"""Packing 12 unit hexagons inside a regular hexagon (min outer side).

Structural design: represent the 12 inner hexagons as a rigid "chain" of
hexagons linked by rotational joints. Optimization variables are the 11
relative rotation angles plus a single global scale factor (12 params total),
drastically smaller than the 36-DoF free placement. We minimize the side
length of the smallest enclosing regular hexagon (fixed orientation) with a
seeded multi-start optimizer. A cheap lattice fallback guarantees a valid
result.
"""
import time
import math
import numpy as np

N = 12
DEADLINE = 135.0          # wall-clock safety margin (evaluator kills at ~150s)
TOL = 1e-9


def _hex_vertices(cx, cy, ang_deg, s=1.0):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    v = np.empty((6, 2))
    for k in range(6):
        t = math.radians(60.0 * k)
        px, py = s * math.cos(t), s * math.sin(t)
        v[k, 0] = cx + px * ca - py * sa
        v[k, 1] = cy + px * sa + py * ca
    return v


def _enclosing_side(pts):
    """Smallest side length of a regular hexagon (fixed orientation, center 0)
    that contains all points in pts (Nx2)."""
    c = np.cos(np.deg2rad([0, 60, 120, 180, 240, 300]))
    s = np.sin(np.deg2rad([0, 60, 120, 180, 240, 300]))
    proj = pts @ np.column_stack((c, s))      # (N,6)
    m = np.max(proj, axis=0)
    M = np.min(proj, axis=0)
    return float(np.max((m + M) / np.sqrt(3.0)))


def _build_chain(theta, scale):
    """theta: (11,) relative joint angles (deg). Returns (centers (12,2),
    absolute angles (12,))."""
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    centers[0] = [0.0, 0.0]
    angles[0] = 0.0
    for i in range(1, N):
        a = angles[i - 1]
        d = math.radians(theta[i - 1])
        # direction of the joint = rotated by (theta-90) from +x, in world frame
        phi = a + d - math.pi / 2.0
        step = 2.0 * scale
        centers[i, 0] = centers[i - 1, 0] + step * math.cos(phi)
        centers[i, 1] = centers[i - 1, 1] + step * math.sin(phi)
        angles[i] = a + theta[i - 1]
    return centers, angles


def _objective(x):
    theta = x[:11]
    scale = max(1e-3, float(x[11]))
    centers, angles = _build_chain(theta, scale)
    pts = np.vstack([_hex_vertices(centers[i, 0], centers[i, 1], angles[i], 1.0)
                     for i in range(N)])
    return _enclosing_side(pts)


def _lattice_fallback():
    """3x4 grid of hexagons (adjacent edges touch). Returns a valid config."""
    cols, rows = 4, 3
    centers = []
    for r in range(rows):
        for c in range(cols):
            x = c * 3.0
            y = r * 2.0 * math.sqrt(3.0)
            centers.append([x, y, 0.0])
    xs = [p[0] for p in centers]
    ys = [p[1] for p in centers]
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    for p in centers:
        p[0] -= cx
        p[1] -= cy
    pts = np.vstack([_hex_vertices(p[0], p[1], p[2], 1.0) for p in centers])
    side = _enclosing_side(pts)
    return centers, (0.0, 0.0), side, 0.0


def _optimize_chain():
    best = (None, None, None, None, None, None)  # inner, center, side, angle, theta, scale
    best_side = float("inf")
    rng = np.random.default_rng(12345)
    t0 = time.time()
    while time.time() < DEADLINE:
        # random start: relative angles in [-150,150], scale in [0.7,1.3]
        theta0 = rng.uniform(-150.0, 150.0, size=11)
        scale0 = rng.uniform(0.7, 1.3)
        x0 = np.concatenate([theta0, [scale0]])
        try:
            res = _scipy_minimize(_objective, x0, method="Nelder-Mead",
                                  options={"maxiter": 400, "xatol": 1e-4,
                                           "fatol": 1e-5, "adaptive": True})
        except Exception:
            res = None
        if res is None or res.fun >= best_side:
            continue
        theta = res.x[:11]
        scale = max(1e-3, float(res.x[11]))
        centers, angles = _build_chain(theta, scale)
        pts = np.vstack([_hex_vertices(centers[i, 0], centers[i, 1], angles[i], 1.0)
                         for i in range(N)])
        side = _enclosing_side(pts)
        # recenter so the enclosing hexagon is centered at origin
        c = np.cos(np.deg2rad([0, 60, 120, 180, 240, 300]))
        s = np.sin(np.deg2rad([0, 60, 120, 180, 240, 300]))
        proj = pts @ np.column_stack((c, s))
        m = np.max(proj, axis=0)
        M = np.min(proj, axis=0)
        # shift in x,y so that each pair of opposite normals is centered
        # solve least-squares for translation
        A = np.column_stack((c, s))          # (6,2)
        b = (m + M) / 2.0                    # (6,) desired center of each normal pair
        shift, *_ = np.linalg.lstsq(A, b, rcond=None)
        for i in range(N):
            centers[i, 0] -= shift[0]
            centers[i, 1] -= shift[1]
        pts2 = np.vstack([_hex_vertices(centers[i, 0], centers[i, 1], angles[i], 1.0)
                          for i in range(N)])
        side2 = _enclosing_side(pts2)
        if side2 < best_side:
            best_side = side2
            inner = [[float(centers[i, 0]), float(centers[i, 1]), float(angles[i])]
                     for i in range(N)]
            best = (inner, (0.0, 0.0), float(side2), 0.0, theta, scale)
    return best


# late import so a missing scipy doesn't kill module load
try:
    from scipy.optimize import minimize as _scipy_minimize
except Exception:
    _scipy_minimize = None


def construct():
    if _scipy_minimize is None:
        return _lattice_fallback()
    best = _optimize_chain()
    if best[0] is None or best[2] >= 40.0:
        return _lattice_fallback()
    inner, center, side, angle = best[0], best[1], best[2], best[3]
    # ensure we never do worse than the seed's trivial value
    if side > 40.0:
        return _lattice_fallback()
    return inner, center, side, angle
