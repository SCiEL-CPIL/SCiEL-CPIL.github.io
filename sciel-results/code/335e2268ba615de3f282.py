# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _score(pts):
    """Return A_min / A_hull for a (13,2) point set; 0.0 if invalid."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (13, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    n = 13
    # All triangle areas, vectorized
    i = np.arange(n)
    ii, jj, kk = np.meshgrid(
        np.arange(n - 2), np.arange(1, n - 1), np.arange(2, n), indexing="ij"
    )
    ii, jj, kk = ii.ravel(), jj.ravel(), kk.ravel()
    p1 = pts[ii]; p2 = pts[jj]; p3 = pts[kk]
    cross = (p1[:, 0] * (p2[:, 1] - p3[:, 1])
             + p2[:, 0] * (p3[:, 1] - p1[:, 1])
             + p3[:, 0] * (p1[:, 1] - p2[:, 1]))
    areas = np.abs(cross) * 0.5
    amin = float(areas.min())
    if not np.isfinite(amin) or amin <= 0.0:
        return 0.0
    try:
        ahull = float(ConvexHull(pts).volume)
    except Exception:
        return 0.0
    if not np.isfinite(ahull) or ahull <= 0.0:
        return 0.0
    return amin / ahull


def _regular_12gon_center():
    """Regular 12-gon (radius 1) plus its center -> 13 points, all on hull."""
    k = np.arange(12)
    ang = 2.0 * np.pi * k / 12.0
    pts = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    pts = np.vstack([pts, np.array([0.0, 0.0])])
    return pts


def heilbronn_convex13() -> np.ndarray:
    """
    Maximize A_min / A_hull for 13 points in a convex region.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates.
    """
    deadline = time.time() + 600.0

    base = _regular_12gon_center()
    best = base.copy()
    best_score = _score(best)

    def neg_score(x):
        return -_score(x)

    rng = np.random.default_rng(12345)
    restarts = 0
    while time.time() < deadline - 45.0:
        restarts += 1
        if restarts == 1:
            x0 = best.copy()
        else:
            # perturb the current best (and occasionally the base) for diversity
            if rng.random() < 0.35:
                x0 = base.copy()
            else:
                x0 = best.copy()
            x0 = x0 + rng.normal(0.0, 0.12, size=x0.shape)

        try:
            res = minimize(neg_score, x0, method="Nelder-Mead",
                           options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-9})
            cand = np.asarray(res.x, dtype=float)
            if cand.shape == (13, 2) and np.all(np.isfinite(cand)):
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best = cand.copy()
        except Exception:
            pass

    return best


# EVOLVE-BLOCK-END
