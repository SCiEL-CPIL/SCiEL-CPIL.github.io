"""Optimized program for Heilbronn convex regions n=14."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import math

N = 14

def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute minimum triangle area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    # Vectorized computation for speed
    # points shape (n, 2)
    # We need to check all i < j < k
    # This is O(n^3), for n=14 it is 14*13*12/6 = 364 triangles. Very fast.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute area of convex hull of points."""
    from scipy.spatial import ConvexHull
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _score(points):
    """Compute score: min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    return min_area / hull_area

def _objective(points_flat):
    """Objective function for minimization: negative score."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _refine(points):
    """Refine a point configuration using local optimization."""
    points_flat = points.flatten()
    try:
        # Use Nelder-Mead for local refinement as it doesn't require gradients
        # and is robust for this non-smooth function
        result = minimize(
            _objective,
            points_flat,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        opt_points = result.x.reshape(-1, 2)
        return opt_points
    except Exception:
        return points

def construct():
    """Construct 14 points to maximize minimum triangle area / convex hull area."""
    
    # Define bounds for the coordinates. 
    # We can normalize to [0, 1] x [0, 1] initially, but the hull area normalization 
    # makes the scale invariant. However, keeping points within a reasonable box 
    # helps the optimizer. Let's use [-1, 1] to allow flexibility.
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Strategy 1: Differential Evolution
    # This is a global optimizer that handles the non-smooth nature of the min-triangle-area function well.
    best_points_de = None
    best_score_de = -1.0
    
    try:
        # Use a seed for reproducibility
        de_result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=200,
            tol=1e-8,
            popsize=15,
            init='sobol' # Sobol sequence provides better coverage than random
        )
        
        if de_result.success or de_result.fun < 0:
            best_points_de = de_result.x.reshape(-1, 2)
            best_score_de = -de_result.fun
    except Exception as e:
        pass
        
    # Strategy 2: Structured Initializations + Local Refinement
    # Heilbronn points often have symmetries. 
    # Let's try starting from a regular polygon + center, or other symmetric arrangements.
    
    candidates = []
    
    # Candidate 1: Regular 14-gon (points on boundary)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radius = 1.0
    pts_reg = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    candidates.append(pts_reg)
    
    # Candidate 2: 7 points on outer circle, 7 points on inner circle (offset)
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi/7
    pts_outer = np.column_stack([1.0 * np.cos(angles_outer), 1.0 * np.sin(angles_outer)])
    pts_inner = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
    pts_two_ring = np.vstack([pts_outer, pts_inner])
    candidates.append(pts_two_ring)
    
    # Candidate 3: Grid-based (4x4 minus 2)
    grid_x = [-0.75, -0.25, 0.25, 0.75]
    grid_y = [-0.75, -0.25, 0.25, 0.75]
    pts_grid = []
    for x in grid_x:
        for y in grid_y:
            if len(pts_grid) < N:
                pts_grid.append([x, y])
    candidates.append(np.array(pts_grid[:N]))
    
    # Candidate 4: Random uniform
    rng = np.random.default_rng(42)
    candidates.append(rng.uniform(-1, 1, size=(N, 2)))
    
    # If DE succeeded, add its result as a candidate for refinement
    if best_points_de is not None:
        candidates.append(best_points_de)
        
    # Refine all candidates
    best_points = None
    best_score = -1.0
    
    for cand in candidates:
        refined = _refine(cand)
        score = _score(refined)
        if score > best_score:
            best_score = score
            best_points = refined.copy()
            
    # Final check: if we have a very good score from DE, ensure we didn't lose it
    if best_points_de is not None:
        score_de = _score(best_points_de)
        if score_de > best_score:
            best_score = score_de
            best_points = best_points_de.copy()
            
    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
# EVOLVE-BLOCK-END
