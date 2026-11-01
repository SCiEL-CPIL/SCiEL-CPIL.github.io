"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

# EVOLVE-BLOCK-START

def _triangle_areas(pts):
    """Compute all triangle areas for N points. Returns array of size C(N,3)."""
    n = len(pts)
    areas = np.empty((n * (n - 1) * (n - 2)) // 6)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross product|
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                areas[idx] = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                idx += 1
    return areas


def _score(pts):
    """Score = min triangle area / convex hull area."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, .volume gives area
        if hull_area <= 0:
            return 0.0
        areas = _triangle_areas(pts)
        min_area = np.min(areas)
        return min_area / hull_area
    except Exception:
        return 0.0


def _objective(x):
    """Negative score for minimization. x is flattened 2D coordinates of N points."""
    pts = x.reshape(N, 2)
    return -_score(pts)


def _regular_ngon(n, radius=1.0):
    """Return vertices of a regular n-gon inscribed in circle of given radius."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def construct():
    deadline = time.time() + 110  # Leave 10s safety margin
    
    # Initial guess: regular 14-gon
    initial_pts = _regular_ngon(N)
    x0 = initial_pts.flatten()
    
    # Bounds: points in a reasonable box around origin
    # Regular 14-gon in unit circle, so use bounds [-2, 2]
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    best_x = x0.copy()
    best_score = _score(initial_pts)
    
    # Try differential evolution with limited iterations
    def objective_with_deadline(x):
        if time.time() > deadline:
            return -best_score  # Return current best to stop search
        return _objective(x)
    
    try:
        result = differential_evolution(
            objective_with_deadline,
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            polish=True,
            init='sobol'
        )
        
        if time.time() < deadline:
            candidate_score = -result.fun
            if candidate_score > best_score:
                best_score = candidate_score
                best_x = result.x.copy()
        
        # Local polish with Nelder-Mead if time permits
        if time.time() < deadline - 5:
            from scipy.optimize import minimize
            res_nm = minimize(
                _objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            nm_score = -res_nm.fun
            if nm_score > best_score:
                best_score = nm_score
                best_x = res_nm.x.copy()
                
    except Exception:
        pass
    
    # Return the best points found
    return best_x.reshape(N, 2)

# EVOLVE-BLOCK-END
