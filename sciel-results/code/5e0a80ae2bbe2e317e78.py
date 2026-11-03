"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time
from scipy.spatial import ConvexHull

def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _triangle_areas(points):
    """Compute all triangle areas."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                areas.append(area)
    return np.array(areas)

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    return np.min(_triangle_areas(points))

def construct():
    """
    Place 13 points in a planar convex region to maximize the minimum triangle area
    divided by the convex hull area.
    """
    best_score = 0.0
    best_points = None
    
    # Initial configurations
    initial_configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    reg_points = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_configs.append(reg_points)
    
    # 2. Grid-like arrangement
    grid_points = []
    for i in range(4):
        for j in range(4):
            if i == 3 and j == 3:
                continue
            grid_points.append([i/3.0, j/3.0])
    grid_points = np.array(grid_points)
    initial_configs.append(grid_points)
    
    # 3. Random points
    rng = np.random.RandomState(42)
    rand_points = rng.rand(N, 2)
    initial_configs.append(rand_points)
    
    # 4. Perturbed regular polygon
    perturbed = reg_points.copy()
    for i in range(N):
        perturbed[i] += rng.rand(2) * 0.1
    initial_configs.append(perturbed)

    # 5. Another random
    rand_points2 = rng.rand(N, 2)
    initial_configs.append(rand_points2)

    # Objective function using log-sum-exp approximation of min area
    beta = 50.0
    
    def objective(x):
        points = x.reshape(-1, 2)
        
        try:
            hull_area = _convex_hull_area(points)
        except Exception:
            return 1e6
            
        if hull_area < 1e-8:
            return 1e6
            
        areas = _triangle_areas(points)
        
        neg_areas = -areas
        max_neg = np.max(neg_areas)
        if not np.isfinite(max_neg):
            return 1e6
            
        log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
        approx_min = -log_sum_exp / beta
        
        penalty = 100.0 * (hull_area - 1.0)**2
        
        return -approx_min + penalty

    # Optimize
    for init_points in initial_configs:
        hull_area = _convex_hull_area(init_points)
        if hull_area > 1e-10:
            init_points = init_points / np.sqrt(hull_area)
        else:
            continue
        
        try:
            result = minimize(objective, init_points.flatten(), method='L-BFGS-B',
                             options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            
            opt_points = result.x.reshape(-1, 2)
            hull_area = _convex_hull_area(opt_points)
            if hull_area > 1e-10:
                opt_points = opt_points / np.sqrt(hull_area)
                min_area = _min_triangle_area(opt_points)
                score = min_area / _convex_hull_area(opt_points)
                
                if score > best_score:
                    best_score = score
                    best_points = opt_points.copy()
                    
        except Exception as e:
            pass

    if best_points is None:
        best_points = reg_points
        hull_area = _convex_hull_area(best_points)
        best_points = best_points / np.sqrt(hull_area)
        best_score = _min_triangle_area(best_points) / _convex_hull_area(best_points)

    return best_points
# EVOLVE-BLOCK-END
