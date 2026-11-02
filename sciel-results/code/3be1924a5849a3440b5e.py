"""Rewritten program for Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 13

# EVOLVE-BLOCK-START

def _min_triangle_area(pts):
    """Compute the minimum area over all triples of points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: compute via shoelace formula on convex hull vertices
        # If hull fails, return a large value to penalize
        return float('inf')


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0 or not np.isfinite(hull_a):
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_a


def construct():
    """Construct 13 points to maximize the minimum triangle area / convex hull area."""
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Precompute all triangle indices
    tri_indices = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    
    def score_fast(pts):
        """Fast score computation using vectorized cross products."""
        # pts: (13, 2)
        # For each triple (i,j,k), compute area
        areas = np.zeros(len(tri_indices))
        for idx, (i, j, k) in enumerate(tri_indices):
            v1 = pts[j] - pts[i]
            v2 = pts[k] - pts[i]
            areas[idx] = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
        min_tri = np.min(areas)
        
        hull_a = _hull_area(pts)
        if hull_a <= 0 or not np.isfinite(hull_a):
            return 0.0
        return min_tri / hull_a
    
    # Strategy: Use a structured initial configuration and then optimize
    # Start with points on a circle (high symmetry)
    
    # Initial configuration: 13 points evenly spaced on a circle
    angles = 2 * np.pi * np.arange(N) / N
    pts = np.column_stack((np.cos(angles), np.sin(angles)))
    best_pts = pts.copy()
    best_score = score_fast(best_pts)
    
    # Local optimization using Nelder-Mead on the 26-dimensional space
    # We'll use scipy.optimize.minimize with Nelder-Mead
    
    from scipy.optimize import minimize
    
    def objective(params):
        """Objective to minimize: negative of the score."""
        pts = params.reshape(-1, 2)
        # Ensure points are distinct (add small regularization if needed)
        s = score_fast(pts)
        return -s
    
    # Run multiple restarts with different initial configurations
    best_score = -np.inf
    best_params = None
    
    # Initial guess: circle
    params0 = best_pts.flatten()
    
    # Try Nelder-Mead optimization
    result = minimize(objective, params0, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    if result.fun < best_score:
        best_score = -result.fun
        best_params = result.x.copy()
    
    # Check time
    if time.time() > deadline - 20:
        pts_out = best_params.reshape(-1, 2)
        return pts_out
    
    # Try more restarts with perturbed initial configurations
    rng = np.random.RandomState(42)
    
    for restart in range(5):
        if time.time() > deadline - 15:
            break
        
        # Perturb the best solution found so far
        if best_params is not None:
            perturbation = rng.randn(N, 2) * 0.1
            params0 = best_params + perturbation.flatten()
        else:
            # Random initial configuration
            params0 = rng.randn(N, 2).flatten()
        
        result = minimize(objective, params0, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if result.fun < best_score:
            best_score = -result.fun
            best_params = result.x.copy()
    
    # Final polish with tighter tolerances
    if best_params is not None and time.time() < deadline - 10:
        result = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
        if result.fun < best_score:
            best_score = -result.fun
            best_params = result.x.copy()
    
    pts_out = best_params.reshape(-1, 2)
    return pts_out

# EVOLVE-BLOCK-END
