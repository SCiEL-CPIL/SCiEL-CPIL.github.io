"""Maximize minimum triangle area over 14 points in a unit-area convex region."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
_TRIS = [(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)]


def _min_triangle_area(pts):
    areas = np.empty(len(_TRIS))
    for idx, (i, j, k) in enumerate(_TRIS):
        a = pts[i]
        b = pts[j] - pts[i]
        c = pts[k] - pts[i]
        areas[idx] = 0.5 * np.abs(b[0] * c[1] - b[1] * c[0])
    return areas.min()


def _convex_hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0


def _score(pts):
    hull = _convex_hull_area(pts)
    if hull <= 0:
        return 0.0
    return _min_triangle_area(pts) / hull


def _normalize_unit_area(pts):
    cx = pts.mean(axis=0)
    p = pts - cx
    s = np.linalg.norm(p, axis=1).mean()
    if s < 1e-12:
        return pts
    p = p / s
    hull = _convex_hull_area(p)
    if hull > 0:
        p = p * np.sqrt(1.0 / hull)
    return p


def _softmin_neg(x, T):
    """x = log-areas (>=0). Returns soft-min of areas (positive, decreasing in T)."""
    m = x.min()
    z = np.exp((x - m) / T)
    return T * np.log(z.sum()) + m


def _objective_flat(flat, T):
    pts = flat.reshape(N, 2)
    p = _normalize_unit_area(pts)
    areas = np.empty(len(_TRIS))
    for idx, (i, j, k) in enumerate(_TRIS):
        b = p[j] - p[i]
        c = p[k] - p[i]
        areas[idx] = 0.5 * np.abs(b[0] * c[1] - b[1] * c[0])
    areas = np.maximum(areas, 1e-15)
    return -_softmin_neg(np.log(areas), T)


def _init_rings():
    pts = np.zeros((N, 2))
    for r in range(N):
        ang = 2 * np.pi * r / N
        rad = 0.35 + 0.6 * (r / max(N - 1, 1))
        pts[r] = [rad * np.cos(ang), rad * np.sin(ang)]
    return pts


def _init_hex():
    pts = []
    s = 0.5
    for row in range(-3, 4):
        for col in range(-3, 4):
            x = col * s + (0.5 * s if row % 2 else 0.0)
            y = row * s * 0.866
            if x * x + y * y <= 2.6:
                pts.append([x, y])
    pts = np.array(pts[:N])
    return pts


def _init_random(rng):
    return np.array([[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(N)])


def construct():
    deadline = time.time() + 130.0
    rng = np.random.RandomState(20240517)

    inits = [
        _init_rings(),
        _init_hex(),
        _init_random(rng),
        _init_rings() + 0.08 * rng.randn(N, 2),
        _init_hex() + 0.08 * rng.randn(N, 2),
        _init_random(rng),
        _init_rings() + 0.2 * rng.randn(N, 2),
        _init_random(rng),
    ]

    # Temperature schedule for soft-min annealing (low = closer to hard min)
    temps = [0.05, 0.03, 0.02, 0.012, 0.007, 0.004, 0.0025, 0.0015, 0.001]

    best_pts = None
    best_val = 0.0

    for init in inits:
        if time.time() > deadline:
            break
        # Normalize initial point
        cur = _normalize_unit_area(init)
        val = _score(cur)
        if val > best_val:
            best_val = val
            best_pts = cur.copy()

        # Anneal with L-BFGS-B from current best
        for T in temps:
            if time.time() > deadline:
                break
            try:
                res = minimize(
                    _objective_flat,
                    cur.flatten(),
                    method="L-BFGS-B",
                    options={"maxiter": 150, "maxfun": 400, "ftol": 1e-12},
                )
                cur = _normalize_unit_area(res.x.reshape(N, 2))
                val = _score(cur)
                if val > best_val:
                    best_val = val
                    best_pts = cur.copy()
            except Exception:
                break

    if best_pts is None:
        best_pts = _normalize_unit_area(_init_rings())

    return best_pts
