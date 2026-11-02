"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    # Use scipy's ConvexHull
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: simple polygon area if all points are on hull (approximate)
        # This is a rough fallback; normally ConvexHull works
        return 0.0

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def construct():
    """
    Place 13 points in a planar convex region to maximize the minimum triangle area
    divided by the convex hull area.
    
    Approach:
    1. Start with a good initial configuration (e.g., regular polygon or grid).
    2. Use optimization to maximize the minimum triangle area subject to convex hull area = 1.
    3. Since the problem is scale-invariant in a certain way, we normalize by hull area.
    
    We'll use a multi-start optimization approach with scipy.optimize.minimize.
    The objective is to maximize min_triangle_area / hull_area, which is equivalent to
    minimizing -min_triangle_area / hull_area.
    """
    best_score = 0.0
    best_points = None
    
    # Try several initial configurations
    initial_configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    reg_points = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_configs.append(reg_points)
    
    # 2. Grid-like arrangement
    grid_points = []
    # Arrange in a 4x4 grid minus one corner, then perturb
    for i in range(4):
        for j in range(4):
            if i == 3 and j == 3:
                continue
            grid_points.append([i/3.0, j/3.0])
    grid_points = np.array(grid_points)
    initial_configs.append(grid_points)
    
    # 3. Random points in unit square
    rng = np.random.RandomState(42)
    rand_points = rng.rand(N, 2)
    initial_configs.append(rand_points)
    
    # 4. Slightly perturbed regular polygon
    perturbed = reg_points.copy()
    for i in range(N):
        perturbed[i] += rng.rand(2) * 0.1
    initial_configs.append(perturbed)
    
    # For each initial config, run optimization
    for init_points in initial_configs:
        # Normalize initial points to have convex hull area 1
        hull_area = _convex_hull_area(init_points)
        if hull_area > 1e-10:
            init_points = init_points / np.sqrt(hull_area)
        else:
            continue
        
        # Define objective function to minimize: -min_triangle_area / hull_area
        def objective(x):
            points = x.reshape(-1, 2)
            hull_area = _convex_hull_area(points)
            if hull_area < 1e-10:
                return 1e6  # Penalize degenerate hulls
            min_area = _min_triangle_area(points)
            score = min_area / hull_area
            return -score
        
        # Bounds: keep points in a reasonable region
        bounds = [(-5, 5)] * (2*N)
        
        # Try different optimization methods
        for method in ['Nelder-Mead', 'Powell']:
            try:
                result = minimize(objective, init_points.flatten(), method=method,
                                 bounds=bounds, options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
                
                if result.success or result.fun < 0:
                    opt_points = result.x.reshape(-1, 2)
                    hull_area = _convex_hull_area(opt_points)
                    if hull_area > 1e-10:
                        min_area = _min_triangle_area(opt_points)
                        score = min_area / hull_area
                        if score > best_score:
                            best_score = score
                            best_points = opt_points.copy()
            except Exception:
                pass
    
    # If optimization didn't improve much, use the best initial config
    if best_points is None or best_score < 0.01:
        # Fall back to regular polygon
        best_points = reg_points
        hull_area = _convex_hull_area(best_points)
        if hull_area > 1e-10:
            best_points = best_points / np.sqrt(hull_area)
        best_score = _min_triangle_area(best_points) / _convex_hull_area(best_points)
    
    # Final normalization: ensure convex hull area is exactly 1
    hull_area = _convex_hull_area(best_points)
    if hull_area > 1e-10:
        best_points = best_points / np.sqrt(hull_area)
    
    return best_points
# EVOLVE-BLOCK-END
