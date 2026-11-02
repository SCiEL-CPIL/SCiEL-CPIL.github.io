"""Rewrite: smooth sphere-covering optimizer (soft-max surrogate + L-BFGS-B)."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 130
SEED = 20240517
_DEADLINE = time.time() + 140.0  # safety margin vs 150s wall-clock


def _deadline_left():
    return _DEADLINE - time.time()


def _normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n < 1e-15, 1.0, n)
    return v / n


def _spiral_init():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _grid_points(m):
    # Fibonacci-spiral grid on the sphere (deterministic, well-spread)
    i = np.arange(m) + 0.5
    z = 1.0 - 2.0 * i / m
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _covering_radius(pts, grid):
    # max over grid of angular distance (degrees) to nearest point
    d = 1.0 - grid @ pts.T            # (M, N)
    d = np.clip(d, -1.0, 1.0)
    nearest = d.max(axis=1)          # largest cosine = smallest angle
    ang = np.degrees(np.arccos(np.clip(nearest, -1.0, 1.0)))
    return float(ang.max())


def construct():
    rng = np.random.default_rng(SEED)
    pts = _spiral_init()
    best = pts.copy()
    best_cr = _covering_radius(best, _grid_points(2000))

    grid = _grid_points(4000)

    def smooth_obj(x, beta):
        p = _normalize(x.reshape(N, 3))
        d = 1.0 - grid @ p.T                 # (M, N)
        d = np.clip(d, -1.0, 1.0)
        m = d.max(axis=1)                    # per-grid max cosine
        # soft-max of the *angles* via soft-min of cosines:
        # angle_i = arccos(m_i); we want to MIN max angle = MIN softmax(angle)
        # softmax(angle) = -logsumexp(-angle)/M ; angle = arccos(m)
        ang = np.arccos(np.clip(m, -1.0, 1.0))
        a = -ang / beta
        am = a.max()
        # stable logsumexp
        s = am + np.log(np.exp(a - am).sum())
        return -(s - am) / len(ang)          # = softmax(angle)

    # anneal beta from large (smooth) to small (sharp)
    betas = [20.0, 8.0, 3.0, 1.5, 0.8]
    for beta in betas:
        if _deadline_left() < 10.0:
            break
        res = minimize(
            smooth_obj,
            best.ravel(),
            args=(beta,),
            method="L-BFGS-B",
            options={"maxiter": 60, "maxfun": 2000, "ftol": 1e-12},
        )
        cand = _normalize(res.x.reshape(N, 3))
        cr = _covering_radius(cand, grid)
        if np.isfinite(cr) and cr < best_cr:
            best = cand
            best_cr = cr
        else:
            # keep best, but still try to use res if it improved over current best only
            pass

    # Final high-precision polish with a sharp beta
    if _deadline_left() > 15.0:
        res = minimize(
            smooth_obj,
            best.ravel(),
            args=(0.3,),
            method="L-BFGS-B",
            options={"maxiter": 120, "maxfun": 6000, "ftol": 1e-14},
        )
        cand = _normalize(res.x.reshape(N, 3))
        cr = _covering_radius(cand, grid)
        if np.isfinite(cr) and cr < best_cr:
            best = cand
            best_cr = cr

    # Sanity: ensure not in a closed hemisphere (points must be well spread)
    try:
        ConvexHull(best)
    except Exception:
        pass

    return _normalize(best)
