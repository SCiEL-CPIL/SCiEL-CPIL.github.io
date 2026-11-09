"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 14

# Precompute triangle indices
TRIANGLE_INDICES = np.array(list(itertools.combinations(range(N), 3)))

def _min_triangle_area(pts):
    """Compute minimum triangle area for N points."""
    # pts: (N, 2)
    # For each triangle (i,j,k), area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    pi = pts[TRIANGLE_INDICES[:, 0]]  # (M, 2)
    pj = pts[TRIANGLE_INDICES[:, 1]]  # (M, 2)
    pk = pts[TRIANGLE_INDICES[:, 2]]  # (M, 2)
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _convex_hull_area(pts):
    """Compute convex hull area using scipy."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use shoelace formula on all points (approximate)
        # Sort by angle around centroid
        centroid = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        sorted_pts = pts[np.argsort(angles)]
        x = sorted_pts[:, 0]
        y = sorted_pts[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _score(pts):
    """Score = min triangle area / convex hull area."""
    min_area = _min_triangle_area(pts)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return -1.0
    return min_area / hull_area

def _negative_score(pts_flat):
    """Negative score for minimization."""
    pts = pts_flat.reshape(N, 2)
    return -_score(pts)

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer before 120s limit
    
    # Start with regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_pts = x0.copy()
    best_score = _score(best_pts)
    
    # Nelder-Mead optimization
    try:
        result = minimize(
            _negative_score,
            x0.flatten(),
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'maxfev': 10000,
                'xatol': 1e-12,
                'fatol': 1e-14,
                'adaptive': True
            }
        )
        
        if time.time() - start_time < time_limit:
            candidate_pts = result.x.reshape(N, 2)
            candidate_score = _score(candidate_pts)
            if candidate_score > best_score:
                best_pts = candidate_pts
                best_score = candidate_score
        
        # If we have time, do a second pass with tighter tolerances
        if time.time() - start_time < time_limit - 20:
            result2 = minimize(
                _negative_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'maxfev': 6000,
                    'xatol': 1e-14,
                    'fatol': 1e-16,
                    'adaptive': True
                }
            )
            candidate_pts = result2.x.reshape(N, 2)
            candidate_score = _score(candidate_pts)
            if candidate_score > best_score:
                best_pts = candidate_pts
                best_score = candidate_score
        
    except Exception:
        pass
    
    return best_pts
