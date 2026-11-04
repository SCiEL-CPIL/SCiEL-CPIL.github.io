"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute all C(14,3)=364 triple indices
_triples = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_IDX_I = _triples[:, 0]
_IDX_J = _triples[:, 1]
_IDX_K = _triples[:, 2]


def _score(points):
    """min triangle area / convex hull area. Vectorized over 364 triples."""
    p = points.reshape(N, 2)
    a = p[_IDX_I]
    b = p[_IDX_J]
    c = p[_IDX_K]
    # area = 0.5*|cross(b-a, c-a)|
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    min_area = 0.5 * np.min(np.abs(cross))
    # convex hull area
    try:
        hull = ConvexHull(p)
        hull_area = hull.volume  # in 2D, volume = area
    except Exception:
        # degenerate: fall back to a large area so score is low
        hull_area = 1.0
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area


def _neg_score(points):
    return -_score(points)


def construct():
    t0 = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    rng = np.random.default_rng(N)

    best_points = None
    best_val = -np.inf

    def update(pts, val):
        nonlocal best_points, best_val
        if val > best_val:
            best_val = val
            best_points = pts.copy()

    # Initial population: a few diverse starts
    starts = []
    # uniform random
    starts.append(rng.random((N, 2)))
    # grid-like
    gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 4), np.linspace(0.1, 0.9, 4))
    grid = np.column_stack([gx.ravel(), gy.ravel()])[:N]
    starts.append(grid)
    # circle
    th = np.linspace(0, 2 * np.pi, N, endpoint=False)
    circ = np.column_stack([0.5 + 0.45 * np.cos(th), 0.5 + 0.45 * np.sin(th)])
    starts.append(circ)

    # Evaluate starts
    for s in starts:
        v = _score(s)
        update(s, v)

    # Differential Evolution global search
    bounds = [(0.0, 1.0)] * (2 * N)
    # Try DE with a reasonable maxiter; we guard by time
    de_maxiter = 60
    de_popsize = 15
    try:
        result = differential_evolution(
            _neg_score,
            bounds=bounds,
            seed=N,
            maxiter=de_maxiter,
            popsize=de_popsize,
            tol=1e-8,
            init='latinhypercube',
            polish=False,
            vectorized=False,
            updating='immediate',
            callback=None,
        )
        if time.time() - t0 > DEADLINE:
            # out of time, return best
            if best_points is not None:
                return best_points
            return np.array([[0.5, 0.5]] * N)
        de_pts = result.x.reshape(N, 2)
        de_val = _score(de_pts)
        update(de_pts, de_val)
    except Exception:
        pass

    # Nelder-Mead local polish on the best point
    if time.time() - t0 < DEADLINE and best_points is not None:
        try:
            nm = minimize(
                _neg_score,
                best_points.reshape(-1),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True},
            )
            nm_pts = nm.x.reshape(N, 2)
            nm_val = _score(nm_pts)
            update(nm_pts, nm_val)
        except Exception:
            pass

    # If still have time, do a second DE round from the best region for more global exploration
    if time.time() - t0 < DEADLINE - 20 and best_points is not None:
        try:
            # narrow bounds around best
            b2 = []
            bp = best_points.reshape(-1)
            for i in range(2 * N):
                lo = max(0.0, bp[i] - 0.25)
                hi = min(1.0, bp[i] + 0.25)
                if lo >= hi:
                    lo, hi = 0.0, 1.0
                b2.append((lo, hi))
            result2 = differential_evolution(
                _neg_score,
                bounds=b2,
                seed=N + 1,
                maxiter=40,
                popsize=12,
                tol=1e-9,
                init='latinhypercube',
                polish=False,
                vectorized=False,
                updating='immediate',
            )
            r2_pts = result2.x.reshape(N, 2)
            r2_val = _score(r2_pts)
            update(r2_pts, r2_val)
            if time.time() - t0 < DEADLINE:
                nm2 = minimize(
                    _neg_score,
                    best_points.reshape(-1),
                    method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True},
                )
                nm2_pts = nm2.x.reshape(N, 2)
                nm2_val = _score(nm2_pts)
                update(nm2_pts, nm2_val)
        except Exception:
            pass

    if best_points is None:
        return np.array([[0.5, 0.5]] * N)
    return best_points
