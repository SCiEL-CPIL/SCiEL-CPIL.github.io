# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial.distance import pdist


def _metric(pts):
    d = pdist(pts)
    if d.size == 0:
        return 0.0
    dmin = d.min()
    dmax = d.max()
    if dmax <= 0.0:
        return 0.0
    return (dmin / dmax) ** 2


def _surrogate(pts, beta):
    d = pdist(pts) ** 2
    dmax = d.max()
    if dmax <= 0.0:
        return 0.0
    d = d / dmax
    return float(np.log(np.sum(np.exp(-beta * d))) - np.log(np.sum(np.exp(beta * d))))


def _normalize(pts):
    c = pts.mean(axis=0)
    p = pts - c
    r = np.linalg.norm(p, axis=1)
    r = np.where(r > 0.0, r, 1.0)
    return p / r[:, None]


def _fibonacci_sphere(n):
    i = np.arange(n)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    th = phi * i
    return np.column_stack((r * np.cos(th), r * np.sin(th), y))


def _seed_3d(n):
    # Fibonacci sphere + golden-ratio vertical stack, then blend toward the sphere
    a = _fibonacci_sphere(n)
    b = np.zeros((n, 3))
    b[:, 2] = 1.0 - 2.0 * np.linspace(0.0, 1.0, n)
    pts = 0.5 * a + 0.5 * b
    return _normalize(pts)


def min_max_dist_dim3_14() -> np.ndarray:
    t0 = time.time()
    deadline = t0 + 570.0
    n = 14

    best_pts = _seed_3d(n)
    best_m = _metric(best_pts)

    # Beta-annealed Nelder-Mead refinement
    betas = [1.5, 3.0, 6.0, 12.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0]
    cur = best_pts.copy()
    for beta in betas:
        if time.time() > deadline - 15.0:
            break
        try:
            from scipy.optimize import minimize
            res = minimize(
                lambda v: _surrogate(v.reshape(n, 3), beta),
                cur.reshape(-1).copy(),
                method="Nelder-Mead",
                options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-12, "adaptive": True},
            )
            cand = _normalize(res.x.reshape(n, 3))
        except Exception:
            cand = cur
        m = _metric(cand)
        if m > best_m:
            best_m = m
            best_pts = cand
        cur = cand

    return best_pts


# EVOLVE-BLOCK-END
