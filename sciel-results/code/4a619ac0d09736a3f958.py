"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from itertools import combinations
import time

def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
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

def _all_triangle_areas(points):
    """Compute areas of all triangles."""
    n = len(points)
    areas = []
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                areas.append(area)
                indices.append((i, j, k))
    return np.array(areas), indices

def _residuals(x, active_indices, target_hull_area=1.0):
    """
    Residuals for least_squares.
    We want to equalize the areas of the active triangles.
    x is a flat array of 2*N coordinates.
    """
    points = x.reshape(-1, 2)
    
    # Check for degenerate hull
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return np.ones(len(active_indices) + 1) * 1e6

    # Compute areas of active triangles
    active_areas = []
    for idx in active_indices:
        i, j, k = idx
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
        active_areas.append(area)
    
    active_areas = np.array(active_areas)
    
    # Residual 1: Deviation of active areas from their mean
    mean_area = np.mean(active_areas)
    res_areas = active_areas - mean_area
    
    # Residual 2: Deviation of hull area from target
    res_hull = hull_area - target_hull_area
    
    return np.concatenate([res_areas, [res_hull]])

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
    
    # 2. Grid-like arrangement (4x4 minus one)
    grid_points = []
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
    
    # 4. Perturbed regular polygon
    perturbed = reg_points.copy()
    for i in range(N):
        perturbed[i] += rng.rand(2) * 0.1
    initial_configs.append(perturbed)
    
    # Try different numbers of active triangles
    K_values = [5, 10, 15, 20]
    
    for init_points in initial_configs:
        # Normalize initial points to have convex hull area 1
        hull_area = _convex_hull_area(init_points)
        if hull_area > 1e-10:
            init_points = init_points / np.sqrt(hull_area)
        else:
            continue
            
        # Compute initial areas to find active triangles
        areas, indices = _all_triangle_areas(init_points)
        
        for K in K_values:
            # Find the K smallest triangles
            sorted_indices = np.argsort(areas)
            active_indices = [indices[i] for i in sorted_indices[:K]]
            
            current_points = init_points.copy()
            
            # Iterative refinement with updating active set
            for outer_iter in range(5):
                def residuals(x):
                    return _residuals(x, active_indices)
                    
                bounds = [(-2, 2)] * (2*N)
                
                try:
                    result = least_squares(residuals, current_points.flatten(), method='trf', bounds=bounds, max_nfev=500)
                    current_points = result.x.reshape(-1, 2)
                except Exception:
                    break
                
                # Update active set
                areas, indices = _all_triangle_areas(current_points)
                sorted_indices = np.argsort(areas)
                new_active_indices = [indices[i] for i in sorted_indices[:K]]
                
                # Check if active set changed significantly
                if set(new_active_indices) == set(active_indices):
                    break
                active_indices = new_active_indices
                
            # Evaluate final score
            hull_area = _convex_hull_area(current_points)
            if hull_area > 1e-10:
                min_area = _min_triangle_area(current_points)
                score = min_area / hull_area
                if score > best_score:
                    best_score = score
                    best_points = current_points.copy()
    
    # If optimization didn't improve much, use the best initial config
    if best_points is None or best_score < 0.01:
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
