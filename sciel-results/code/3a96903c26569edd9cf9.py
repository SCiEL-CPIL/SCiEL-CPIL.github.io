"""Optimized program for Heilbronn convex regions n=14."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

# Precompute all triples
_TRIPLES = list(itertools.combinations(range(N), 3))
_TRIPLES_ARR = np.array(_TRIPLES, dtype=np.int32)


def _all_triangle_areas(points):
    """Compute all triangle areas for N points using vectorized ops.
    Returns array of shape (C(N,3),).
    """
    n = len(points)
    # points: (n, 2)
    # For each triple (i, j, k): area = 0.5 * |cross(pj-pi, pk-pi)|
    pi = points[_TRIPLES_ARR[:, 0]]  # (M, 2)
    pj = points[_TRIPLES_ARR[:, 1]]  # (M, 2)
    pk = points[_TRIPLES_ARR[:, 2]]  # (M, 2)
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _convex_hull_area(points):
    """Compute area of convex hull of points."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, .volume is area
    except Exception:
        # Fallback: compute via shoelace on sorted points
        # If all points collinear, area is 0
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    return min_area / hull_area


def _neg_score(points):
    """Negative score for minimization."""
    return -_score(points)


def _initial_points(seed):
    """Generate initial points: regular heptagon-like structure with perturbations."""
    rng = np.random.RandomState(seed)
    # Start with a regular 14-gon on a circle, then perturb
    angles = 2 * np.pi * np.arange(N) / N
    radii = 1.0 + 0.1 * rng.randn(N)
    radii = np.clip(radii, 0.3, 1.5)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    # Add small random perturbation
    points += 0.05 * rng.randn(N, 2)
    return points


def _local_search(points, time_limit=10.0):
    """Local search using L-BFGS-B on the 28 coordinates."""
    start_time = time.time()
    x0 = points.flatten()
    
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Try multiple random restarts of local optimization
    n_restarts = 3
    for r in range(n_restarts):
        if time.time() - start_time > time_limit - 5.0:
            break
        if r == 0:
            x0 = points.flatten()
        else:
            # Perturb the best found so far
            perturbation = 0.1 * np.random.randn(2 * N)
            x0 = best_points.flatten() + perturbation
        
        try:
            result = minimize(
                _neg_score,
                x0,
                method='L-BFGS-B',
                jac=None,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            new_points = result.x.reshape(N, 2)
            new_score = _score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
        except Exception:
            continue
    
    return best_points, best_score


def _refine_with_soft_min(points, time_limit=15.0):
    """Refine using a smooth approximation of the minimum to help gradient-based methods."""
    start_time = time.time()
    x0 = points.flatten()
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Use a soft-min (log-sum-exp) approximation for smoother optimization
    def _soft_neg_score(x):
        pts = x.reshape(N, 2)
        areas = _all_triangle_areas(pts)
        hull_area = _convex_hull_area(pts)
        if hull_area <= 1e-12:
            return 1e6
        # Soft min: -log(sum(exp(-areas/T))) / T approximates min(areas)
        T = 0.01
        log_sum = np.log(np.sum(np.exp(-areas / T)))
        soft_min = -T * log_sum
        return -(soft_min / hull_area)
    
    try:
        result = minimize(
            _soft_neg_score,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        new_points = result.x.reshape(N, 2)
        new_score = _score(new_points)
        if new_score > best_score:
            best_score = new_score
            best_points = new_points
    except Exception:
        pass
    
    return best_points, best_score


def construct():
    """Construct 14 points to maximize min triangle area / convex hull area."""
    deadline = 120.0
    start_time = time.time()
    
    best_points = None
    best_score = -np.inf
    
    # Multi-start local search
    n_starts = 8
    for i in range(n_starts):
        if time.time() - start_time > deadline - 20.0:
            break
        
        seed = 42 + i * 137
        pts = _initial_points(seed)
        
        # Local search phase
        pts, sc = _local_search(pts, time_limit=min(10.0, max(1.0, deadline - 20.0 - (time.time() - start_time))))
        
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
        
        # Check time
        if time.time() - start_time > deadline - 15.0:
            break
    
    # Refine the best solution with soft-min approach
    if best_points is not None and time.time() - start_time < deadline - 10.0:
        refined_pts, refined_sc = _refine_with_soft_min(
            best_points,
            time_limit=min(15.0, deadline - 10.0 - (time.time() - start_time))
        )
        if refined_sc > best_score:
            best_score = refined_sc
            best_points = refined_pts.copy()
    
    # Final local polish if time permits
    if best_points is not None and time.time() - start_time < deadline - 5.0:
        polished_pts, polished_sc = _local_search(
            best_points,
            time_limit=min(5.0, deadline - 5.0 - (time.time() - start_time))
        )
        if polished_sc > best_score:
            best_score = polished_sc
            best_points = polished_pts.copy()
    
    if best_points is None:
        # Fallback: regular 14-gon
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
