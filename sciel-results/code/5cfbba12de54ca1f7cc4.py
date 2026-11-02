"""Heilbronn convex regions: maximize min triangle area over convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

_DEADLINE = 150.0
_START = time.time()
_SAFETY = 10.0  # seconds before deadline to stop


def _min_triangle_area(pts):
    """Minimum area of any triangle formed by 3 of the points."""
    n = len(pts)
    best = np.inf
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < best:
                    best = area
    return best


def _convex_hull_area(pts):
    """Area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # in 2D, volume == area
    except Exception:
        # Fallback: shoelace on sorted points
        order = np.lexsort((pts[:, 1], pts[:, 0]))
        p = pts[order]
        x, y = p[:, 0], p[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _score(pts):
    """min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    return _min_triangle_area(pts) / hull_area


def _regular_polygon_init(n, center=(0.5, 0.5), radius=0.4):
    """Place n points: regular (n-1)-gon around center + 1 center point."""
    m = n - 1
    angles = 2 * np.pi * np.arange(m) / m
    pts = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])
    pts = np.vstack([pts, np.array(center)])
    return pts


def construct():
    global _START, _DEADLINE
    elapsed = time.time() - _START
    remaining = _DEADLINE - _SAFETY - elapsed
    if remaining < 5.0:
        # Not enough time, return quick heuristic
        return _regular_polygon_init(N)

    # Start with regular polygon configuration
    best_pts = _regular_polygon_init(N)
    best_score = _score(best_pts)

    # Try multiple starting configurations
    inits = [
        _regular_polygon_init(N, radius=0.4),
        _regular_polygon_init(N, radius=0.45),
        _regular_polygon_init(N, radius=0.35),
    ]
    # Also try a slightly perturbed version
    rng = np.random.RandomState(42)
    for _ in range(3):
        base = _regular_polygon_init(N, radius=0.4)
        perturbed = base + rng.randn(N, 2) * 0.02
        inits.append(perturbed)

    # Evaluate all inits
    for init in inits:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init.copy()

    # Nelder-Mead optimization on the best starting point
    # Flatten points to 1D vector for scipy.optimize
    x0 = best_pts.flatten()

    def _neg_score(x):
        pts = x.reshape(N, 2)
        return -_score(pts)

    # Run Nelder-Mead with time checking
    options = {
        'maxiter': 500,
        'xatol': 1e-6,
        'fatol': 1e-8,
        'adaptive': True,
    }

    # Use callback to check time
    time_check = {'stop': False}

    def _callback(xk):
        if time.time() - _START > _DEADLINE - _SAFETY:
            time_check['stop'] = True
        return time_check['stop']

    try:
        result = minimize(_neg_score, x0, method='Nelder-Mead',
                          options=options, callback=_callback)
        if not time_check['stop'] and result.fun < -best_score:
            candidate = result.x.reshape(N, 2)
            s = _score(candidate)
            if s > best_score:
                best_score = s
                best_pts = candidate.copy()
    except Exception:
        pass

    # If time remains, try a second round of optimization from a different start
    if time.time() - _START < _DEADLINE - _SAFETY - 30:
        # Try optimizing from a different initial configuration
        alt_init = _regular_polygon_init(N, radius=0.42)
        x0_alt = alt_init.flatten()
        time_check['stop'] = False
        try:
            result2 = minimize(_neg_score, x0_alt, method='Nelder-Mead',
                               options={'maxiter': 400, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True},
                               callback=_callback)
            if not time_check['stop']:
                candidate2 = result2.x.reshape(N, 2)
                s2 = _score(candidate2)
                if s2 > best_score:
                    best_score = s2
                    best_pts = candidate2.copy()
        except Exception:
            pass

    # Final: normalize so the convex hull has unit area (the task says "normalized to unit area")
    # Actually, re-reading: "score is the minimum triangle area divided by the convex hull area"
    # So we just return the points as-is; the evaluator computes the ratio.
    # But let's make sure points are in a reasonable range.
    return best_pts
# EVOLVE-BLOCK-END
