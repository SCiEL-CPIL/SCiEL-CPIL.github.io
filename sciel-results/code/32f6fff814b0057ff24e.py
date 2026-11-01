"""Improved program: smooth log-sum-exp surrogate + Nelder-Mead polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 120.0
SAFETY = 15.0  # seconds to keep before the 120s wall-clock kill


def _pairwise_dist_sq(pts):
    """Return (n, n) squared distance matrix."""
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sum(diff * diff, axis=2)


def _radius_from_centers(pts):
    """Compute the raw score: min half-pair-distance and half-wall-distance."""
    d2 = _pairwise_dist_sq(pts)
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(d2.min()) / 2.0
    wall = 0.5 - np.abs(pts).max(axis=0)  # per-axis max |coord|
    min_wall = 0.5 - np.abs(pts).max()
    return min(min_pair, min_wall)


def _soft_radius(pts, beta):
    """Smooth surrogate: soft-min over pair distances and walls.
    Returns a scalar to MAXIMIZE (use negative for minimizers)."""
    pts = np.asarray(pts).reshape(-1, 3)
    d2 = _pairwise_dist_sq(pts)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    # soft-min of pair distances: -logsumexp(-d * beta) / beta  -> approx min d
    # use log-sum-exp on negative distances
    neg_d = -d * beta
    m = neg_d.max()
    soft_min_pair = -(m + np.log(np.sum(np.exp(neg_d - m)))) / beta
    # walls: for each point, distance to nearest wall = 0.5 - max|coord|
    # soft-min over all points and all 6 walls
    # wall distances per point: 0.5 - |x|, 0.5 + x, 0.5 - |y|, ...
    ax = pts[:, 0]
    ay = pts[:, 1]
    az = pts[:, 2]
    wall_d = np.stack([
        0.5 - np.abs(ax), 0.5 - np.abs(ay), 0.5 - np.abs(az),
        0.5 + ax, 0.5 + ay, 0.5 + az,
    ], axis=1)  # (N, 6)
    # clip tiny negatives for log safety
    wall_d = np.clip(wall_d, 1e-12, None)
    neg_w = -wall_d * beta
    mw = neg_w.max()
    soft_min_wall = -(mw + np.log(np.sum(np.exp(neg_w - mw)))) / beta
    return min(soft_min_pair, soft_min_wall)


def _neg_soft(pts, beta):
    return -_soft_radius(pts, beta)


def _time_left():
    return DEADLINE - SAFETY - (time.time() - _t0)


def _nm_polish(pts):
    """Local polish using Nelder-Mead on the true (non-smooth) objective."""
    def obj(x):
        return -_radius_from_centers(x.reshape(-1, 3))
    try:
        res = minimize(obj, pts.reshape(-1), method='Nelder-Mead',
                       options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12})
        return res.x.reshape(-1, 3)
    except Exception:
        return pts


def _lgbf_smooth(pts, beta):
    """L-BFGS-B on the smooth surrogate."""
    def obj(x):
        return _neg_soft(x.reshape(-1, 3), beta)
    def bnd():
        return [(-0.5, 0.5)] * (3 * N)
    try:
        res = minimize(obj, pts.reshape(-1), method='L-BFGS-B',
                       bounds=bnd(),
                       options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-8})
        return res.x.reshape(-1, 3)
    except Exception:
        return pts


def _grid_init(m=4):
    """Simple cubic grid, take first N points."""
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _fcc_init():
    """FCC-like lattice init: two interpenetrating simple cubes."""
    s = 0.5 / 3.0
    pts = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                pts.append((i * s, j * s, k * s))
                pts.append((i * s + s / 2, j * s + s / 2, k * s + s / 2))
    pts = np.array(pts)
    # keep those inside the cube
    mask = np.all(np.abs(pts) <= 0.5, axis=1)
    pts = pts[mask]
    if len(pts) < N:
        # pad with jittered grid
        g = _grid_init(4)
        pts = np.vstack([pts, g])
    return pts[:N].copy()


def _random_init(rng):
    return (rng.random((N, 3)) - 0.5) * 0.9


def _make_inits(rng):
    inits = []
    inits.append(_grid_init(4))
    inits.append(_fcc_init())
    for _ in range(4):
        inits.append(_random_init(rng))
    return inits


def _clamp(pts):
    return np.clip(pts, -0.5, 0.5)


def construct():
    global _t0
    _t0 = time.time()
    rng = np.random.default_rng(20240501)

    inits = _make_inits(rng)
    best_pts = None
    best_r = -1.0

    # Phase 1: smooth surrogate with annealed beta, then polish each.
    betas = [20.0, 60.0, 150.0]
    for init in inits:
        if _time_left() < 8.0:
            break
        cur = init.copy()
        for beta in betas:
            if _time_left() < 8.0:
                break
            cur = _lgbf_smooth(cur, beta)
            cur = _clamp(cur)
        # polish
        if _time_left() < 5.0:
            break
        cur = _nm_polish(cur)
        cur = _clamp(cur)
        r = _radius_from_centers(cur)
        if r > best_r:
            best_r = r
            best_pts = cur.copy()

    # Ensure we always have a valid fallback.
    if best_pts is None:
        best_pts = _grid_init(4)
        best_r = _radius_from_centers(best_pts)

    # Final polish of the best.
    if _time_left() > 5.0:
        polished = _nm_polish(best_pts)
        polished = _clamp(polished)
        rp = _radius_from_centers(polished)
        if rp > best_r:
            best_r = rp
            best_pts = polished

    # Extra random-restart polish budget if time remains.
    extra = 0
    while _time_left() > 12.0 and extra < 12:
        init = _random_init(rng)
        cur = init.copy()
        for beta in betas:
            if _time_left() < 12.0:
                break
            cur = _lgbf_smooth(cur, beta)
            cur = _clamp(cur)
        if _time_left() < 5.0:
            break
        cur = _nm_polish(cur)
        cur = _clamp(cur)
        r = _radius_from_centers(cur)
        if r > best_r:
            best_r = r
            best_pts = cur.copy()
        extra += 1

    return best_pts
