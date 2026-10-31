"""Circle packing in unit square - Voronoi-based optimization"""
import numpy as np
from scipy.spatial import Delaunay, Voronoi
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 140  # Safety margin before 150s limit

def _compute_voronoi_radii(points):
    """Compute the maximum radius for each point based on Voronoi cell distances."""
    n = len(points)
    if n < 3:
        # Degenerate case
        radii = np.zeros(n)
        for i in range(n):
            x, y = points[i]
            radii[i] = min(x, 1-x, y, 1-y)
        return radii
    
    try:
        vor = Voronoi(points)
    except Exception:
        # Fallback: use minimum distance to boundaries and other points
        radii = np.zeros(n)
        for i in range(n):
            x, y = points[i]
            min_dist = min(x, 1-x, y, 1-y)
            for j in range(n):
                if i != j:
                    d = np.sqrt((x-points[j][0])**2 + (y-points[j][1])**2)
                    min_dist = min(min_dist, d/2)
            radii[i] = max(0, min_dist - 1e-10)
        return radii
    
    radii = np.zeros(n)
    for i in range(n):
        x, y = points[i]
        # Distance to boundaries
        min_dist = min(x, 1-x, y, 1-y)
        
        # Check Voronoi edges
        region = vor.regions[vor.point_region[i]]
        if -1 not in region and len(region) > 0:
            for edge_idx in vor.ridge_points[vor.point_region[i]]:
                pass  # Complex, use simpler approach
        
        # Simpler: distance to nearest other point / 2, and to boundaries
        for j in range(n):
            if i != j:
                d = np.sqrt((x-points[j][0])**2 + (y-points[j][1])**2)
                min_dist = min(min_dist, d/2)
        
        radii[i] = max(0, min_dist - 1e-10)
    
    return radii

def _objective_and_grad(points_flat):
    """Objective: negative sum of radii (to minimize). Returns value and gradient."""
    points = points_flat.reshape(-1, 2)
    n = len(points)
    
    # Compute radii
    radii = np.zeros(n)
    for i in range(n):
        x, y = points[i]
        min_dist = min(x, 1-x, y, 1-y)
        for j in range(n):
            if i != j:
                d = np.sqrt((x-points[j][0])**2 + (y-points[j][1])**2)
                min_dist = min(min_dist, d/2)
        radii[i] = max(0, min_dist - 1e-10)
    
    total = np.sum(radii)
    
    # Gradient computation (numerical for simplicity, could be analytical)
    grad = np.zeros_like(points_flat)
    eps = 1e-7
    for k in range(len(points_flat)):
        pf = points_flat.copy()
        pf[k] += eps
        p_plus = pf.reshape(-1, 2)
        r_plus = np.zeros(n)
        for i in range(n):
            x, y = p_plus[i]
            min_dist = min(x, 1-x, y, 1-y)
            for j in range(n):
                if i != j:
                    d = np.sqrt((x-p_plus[j][0])**2 + (y-p_plus[j][1])**2)
                    min_dist = min(min_dist, d/2)
            r_plus[i] = max(0, min_dist - 1e-10)
        grad[k] = (np.sum(r_plus) - total) / eps
    
    return -total, grad

def _hexagonal_init(n):
    """Generate initial hexagonal packing configuration."""
    points = []
    rows = int(np.ceil(np.sqrt(n * 0.866)))  # Hexagonal spacing
    cols = int(np.ceil(n / rows))
    
    row_height = 1.0 / (rows + 1)
    col_width = 1.0 / (cols + 1)
    
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= n:
                break
            x = (c + 1) * col_width
            y = (r + 1) * row_height
            if r % 2 == 1:
                x += col_width * 0.5
            if 0.01 <= x <= 0.99 and 0.01 <= y <= 0.99:
                points.append([x, y])
                count += 1
    
    # If we didn't get enough, add random points
    while len(points) < n:
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        points.append([x, y])
    
    return np.array(points[:n])

def construct():
    start_time = time.time()
    
    # Initial configuration
    np.random.seed(42)
    points = _hexagonal_init(N)
    
    # Gradient ascent optimization
    best_points = points.copy()
    best_score = -np.sum(_compute_voronoi_radii(points))
    
    # Use scipy minimize with numerical gradient
    x0 = points.flatten()
    
    result = minimize(
        _objective_and_grad,
        x0,
        method='L-BFGS-B',
        bounds=[(0.01, 0.99)] * (2*N),
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    if time.time() - start_time > DEADLINE:
        # Return best found so far
        pass
    else:
        if result.success or result.fun < best_score:
            best_points = result.x.reshape(-1, 2)
            best_score = result.fun
    
    # Compute final radii
    radii = _compute_voronoi_radii(best_points)
    
    # Build result array
    result_arr = np.zeros((N, 3))
    for i in range(N):
        result_arr[i] = [best_points[i][0], best_points[i][1], radii[i]]
    
    return result_arr
