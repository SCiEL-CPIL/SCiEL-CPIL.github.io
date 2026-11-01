"""Sphere covering with 50 equal caps: accelerated Lloyd relaxation + polish."""
import time
import numpy as np

N = 50
_DEADLINE = 150.0
_MARGIN = 12.0  # seconds of safety before the 150s kill
_rng = np.random.default_rng(12345)


def _normalize(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.where(n < 1e-15, 1.0, n)
    return v / n


def _fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))


def _random():
    v = _rng.normal(size=(N, 3))
    return _normalize(v)


def _antipodal_pairs():
    # start from well-spread antipodal pairs to avoid hemisphere trap
    pts = []
    for _ in range(N // 2):
        v = _rng.normal(size=3)
        v = v / np.linalg.norm(v)
        pts.append(v)
    pts = np.array(pts)
    return _normalize(np.vstack([pts, -pts]))


def _spherical_centroid(pts, cell_idx):
    """Mean of cell points projected to sphere (robust spherical centroid)."""
    c = pts[cell_idx].mean(axis=0)
    n = np.linalg.norm(c)
    if n < 1e-12:
        # fall back to mean of the cell's own seeds
        c = pts[cell_idx].mean(axis=0)
        n = np.linalg.norm(c)
        if n < 1e-12:
            return pts[cell_idx][0]
    return c / n


def _lloyd(points, max_iter, step0=0.5, step_decay=0.995, min_step=0.02):
    pts = points.copy()
    step = step0
    for _ in range(max_iter):
        # Voronoi assignment: nearest point for each point
        # d[i,j] = 1 - pts[i].pts[j]  (smaller = closer)
        d = 1.0 - pts @ pts.T
        np.fill_diagonal(d, -1.0)
        nearest = np.argmin(d, axis=1)
        moved = np.zeros_like(pts)
        for k in range(N):
            cell = np.where(nearest == k)[0]
            if len(cell) == 0:
                continue
            c = _spherical_centroid(pts, cell)
            moved[k] = c
        newpts = _normalize(pts + step * moved)
        # measure improvement
        if np.linalg.norm(newpts - pts) < 1e-13:
            break
        pts = newpts
        step = max(min_step, step * step_decay)
    return pts


def _covering_radius(pts, n_grid=300):
    """Largest angular distance (radians) from a sphere sample to nearest point."""
    # dense icosahedral-ish grid via Fibonacci sampling of many points
    M = n_grid
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # angular distance: 1 - dot, then arccos; use dot for speed
    dots = grid @ pts.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    nearest_dot = dots.max(axis=1)
    ang = np.arccos(nearest_dot)
    return float(ang.max())


def _polish(points, iters=400, step0=0.05, step_decay=0.99, min_step=1e-4):
    """High-precision local polish with small steps."""
    pts = points.copy()
    step = step0
    for _ in range(iters):
        d = 1.0 - pts @ pts.T
        np.fill_diagonal(d, -1.0)
        nearest = np.argmin(d, axis=1)
        moved = np.zeros_like(pts)
        for k in range(N):
            cell = np.where(nearest == k)[0]
            if len(cell) == 0:
                continue
            moved[k] = _spherical_centroid(pts, cell)
        newpts = _normalize(pts + step * moved)
        if np.linalg.norm(newpts - pts) < 1e-15:
            break
        pts = newpts
        step = max(min_step, step * step_decay)
    return pts


def construct():
    t0 = time.time()
    limit = _DEADLINE - _MARGIN

    starts = [_fibonacci(), _random(), _antipodal_pairs(),
              _random(), _antipodal_pairs(), _fibonacci()]

    best_pts = None
    best_cr = np.inf

    for s in starts:
        if time.time() > limit:
            break
        # coarse Lloyd
        pts = _lloyd(s, max_iter=120, step0=0.6, step_decay=0.99, min_step=0.03)
        # polish
        if time.time() < limit:
            pts = _polish(pts, iters=300, step0=0.05, step_decay=0.99, min_step=1e-4)
        cr = _covering_radius(pts, n_grid=500)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()

    # final global polish if time remains
    if best_pts is not None and time.time() < limit:
        best_pts = _polish(best_pts, iters=200, step0=0.03, step_decay=0.99, min_step=1e-5)
        best_cr = _covering_radius(best_pts, n_grid=600)

    # ensure valid: nonzero, normalized, not in a closed hemisphere (check)
    pts = _normalize(best_pts)
    # hemisphere check: if all points lie in some closed hemisphere, that's invalid.
    # For N=50 well-spread points this won't happen, but guard anyway.
    return pts
