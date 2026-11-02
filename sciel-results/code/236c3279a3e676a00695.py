"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 120.0  # seconds, with safety margin

# EVOLVE-BLOCK-START
import random
import numpy as np


def _triangle_areas(points):
    """Compute all triangle areas for N points. Returns array of C(N,3) areas."""
    n = len(points)
    areas = np.empty((n * (n-1) * (n-2)) // 6)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                areas[idx] = area
                idx += 1
    return areas


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = np.min(_triangle_areas(points))
    return min_tri / hull_area


def _negative_score(params):
    """Objective for optimization: negative of score."""
    points = params.reshape(-1, 2)
    return -_score(points)


def _initial_circle(n, r=1.0):
    """Place n points on a circle of radius r centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])


def _regular_polygon(n, r=1.0):
    """Place n points as regular polygon vertices."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])


def construct():
    start_time = time.time()
    best_points = _regular_polygon(N)
    best_score = _score(best_points)
    
    # Try a few initial configurations
    initial_candidates = [
        _regular_polygon(N),
        _circle(N, 1.0),
        _circle(N, 1.0) + np.random.default_rng(42).normal(0, 0.1, (N, 2)),
    ]
    
    for cand in initial_candidates:
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_points = cand.copy()
    
    # Use differential evolution for global optimization
    # Parameter space: 2N coordinates in [-2, 2]
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    elapsed = time.time() - start_time
    if elapsed < 60:  # Only run DE if we have time
        try:
            result = differential_evolution(
                _negative_score,
                bounds=bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=False,
                init='sobol',
                workers=1
            )
            elapsed = time.time() - start_time
            if elapsed < DEADLINE - 10:
                candidate = result.x.reshape(-1, 2)
                s = _score(candidate)
                if s > best_score:
                    best_score = s
                    best_points = candidate.copy()
        except Exception:
            pass
    
    # Local polish with L-BFGS-B
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 15:
        try:
            local_result = minimize(
                _negative_score,
                best_points.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            elapsed = time.time() - start_time
            if elapsed < DEADLINE - 5:
                candidate = local_result.x.reshape(-1, 2)
                s = _score(candidate)
                if s > best_score:
                    best_score = s
                    best_points = candidate.copy()
        except Exception:
            pass
    
    # Final deadline check
    if time.time() - start_time > DEADLINE - 2:
        pass  # Just return best found
    
    return best_points
# EVOLVE-BLOCK-END
