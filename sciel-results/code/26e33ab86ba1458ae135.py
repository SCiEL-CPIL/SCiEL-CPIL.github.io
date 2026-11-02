"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 13

# EVOLVE-BLOCK-START

def _score_points(pts):
    """Compute min triangle area / convex hull area for a set of points."""
    n = len(pts)
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    # Compute all triangle areas using cross product
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area


def _neg_score(params):
    """Negative score for minimization."""
    pts = params.reshape((N, 2))
    return -_score_points(pts)


def _grid_search_1_6_6():
    """
    Search over a symmetric 1-6-6 configuration:
    - 1 center point
    - 6 points on an inner hexagon (radius r1)
    - 6 points on an outer hexagon (radius r2)
    with optional angular offset between the two hexagons.
    """
    best_score = 0.0
    best_params = None
    
    # Grid over radii and angle offset
    r1_vals = np.linspace(0.1, 0.6, 30)
    r2_vals = np.linspace(0.6, 1.0, 30)
    angle_vals = np.linspace(0, np.pi/6, 15)  # Offset from 0 to 30 degrees
    
    for r1 in r1_vals:
        for r2 in r2_vals:
            for angle in angle_vals:
                pts = np.zeros((13, 2))
                # Center
                pts[0] = [0, 0]
                # Inner hexagon
                for i in range(6):
                    theta = i * np.pi / 3
                    pts[1 + i] = [r1 * np.cos(theta), r1 * np.sin(theta)]
                # Outer hexagon with angle offset
                for i in range(6):
                    theta = i * np.pi / 3 + angle
                    pts[7 + i] = [r2 * np.cos(theta), r2 * np.sin(theta)]
                
                score = _score_points(pts)
                if score > best_score:
                    best_score = score
                    best_params = pts.copy()
    
    return best_score, best_params


def _refine_local(params):
    """Local refinement using Nelder-Mead or similar."""
    try:
        result = minimize(_neg_score, params, method='Nelder-Mead', 
                         options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
        refined_pts = result.x.reshape((N, 2))
        score = _score_points(refined_pts)
        if score > 0:
            return score, refined_pts
    except Exception:
        pass
    return None, None


def _differential_evolution_search():
    """Global search using differential evolution."""
    # Bounds: points in [-1, 1] x [-1, 1]
    bounds = [(-1, 1)] * (2 * N)
    
    deadline = time.time() + 100  # Leave margin for other operations
    
    best_score = 0.0
    best_params = None
    
    def callback(xk, convergence):
        nonlocal best_score, best_params
        if time.time() > deadline:
            raise KeyboardInterrupt
        pts = xk.reshape((N, 2))
        score = _score_points(pts)
        if score > best_score:
            best_score = score
            best_params = pts.copy()
    
    try:
        result = differential_evolution(_neg_score, bounds, maxiter=200, 
                                         popsize=20, tol=1e-6, 
                                         callback=callback, seed=42)
        if result.x is not None:
            pts = result.x.reshape((N, 2))
            score = _score_points(pts)
            if score > best_score:
                best_score = score
                best_params = pts.copy()
    except KeyboardInterrupt:
        pass
    except Exception:
        pass
    
    return best_score, best_params


def construct():
    """Construct 13 points to maximize min triangle area / convex hull area."""
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for safety
    
    best_score = 0.0
    best_pts = None
    
    # Strategy 1: Grid search over symmetric 1-6-6 configuration
    if time.time() < deadline - 60:
        score1, pts1 = _grid_search_1_6_6()
        if pts1 is not None and score1 > best_score:
            best_score = score1
            best_pts = pts1
    
    # Strategy 2: Local refinement of best grid search result
    if best_pts is not None and time.time() < deadline - 20:
        score2, pts2 = _refine_local(best_pts.flatten())
        if pts2 is not None and score2 > best_score:
            best_score = score2
            best_pts = pts2
    
    # Strategy 3: Differential evolution for global search
    if time.time() < deadline - 10:
        score3, pts3 = _differential_evolution_search()
        if pts3 is not None and score3 > best_score:
            best_score = score3
            best_pts = pts3
    
    # If no good result found, use a default configuration
    if best_pts is None:
        # Regular 13-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts

# EVOLVE-BLOCK-END
