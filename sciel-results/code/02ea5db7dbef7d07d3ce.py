"""Heilbronn triangles: maximize minimum triangle area for 11 points in equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11
h = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, h])

def project_to_triangle(p):
    """Project points into the equilateral triangle (0,0), (1,0), (0.5, h)."""
    p = p.copy()
    # Clip to bounding box
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, h)
    
    # Project to left edge: y <= sqrt(3) * x  =>  sqrt(3)*x - y >= 0
    # The line AC: from (0,0) to (0.5, h). Equation: y = sqrt(3) x
    # For a point (x,y), if y > sqrt(3)*x, it's outside on the left.
    # Normal pointing inward: (sqrt(3), -1) normalized by 2.
    dist_left = (np.sqrt(3) * p[:, 0] - p[:, 1]) / 2.0
    mask_left = dist_left < 0
    if np.any(mask_left):
        p[mask_left, 0] += dist_left[mask_left] * np.sqrt(3)
        p[mask_left, 1] -= dist_left[mask_left]
    
    # Project to right edge: y <= -sqrt(3) * (x - 1)  =>  sqrt(3)*(1-x) - y >= 0
    dist_right = (np.sqrt(3) * (1.0 - p[:, 0]) - p[:, 1]) / 2.0
    mask_right = dist_right < 0
    if np.any(mask_right):
        p[mask_right, 0] -= dist_right[mask_right] * np.sqrt(3)
        p[mask_right, 1] -= dist_right[mask_right]
    
    # Final clip to be safe
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, h)
    return p

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the N points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def objective_flat(x):
    """Objective function for optimization: negative of min triangle area."""
    points = project_to_triangle(x.reshape(-1, 2))
    return -min_triangle_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_val = 0.0
    
    # Try multiple random restarts with Nelder-Mead
    rng = np.random.RandomState(42)
    
    # Initial good starting points: use a grid-like distribution
    # The known optimal configuration for 11 points is roughly a hexagonal lattice
    # Let's start with a few different initial configurations
    
    # Configuration 1: Roughly hexagonal
    base_points = [
        [0.5, 0.288675],  # center-ish
        [0.25, 0.144338],
        [0.75, 0.144338],
        [0.125, 0.072169],
        [0.375, 0.072169],
        [0.625, 0.072169],
        [0.875, 0.072169],
        [0.125, 0.216506],
        [0.375, 0.216506],
        [0.625, 0.216506],
        [0.875, 0.216506],
    ]
    base_points = np.array(base_points)
    base_points = project_to_triangle(base_points)
    
    val = min_triangle_area(base_points)
    best_points = base_points.copy()
    best_val = val
    
    # Try Nelder-Mead optimization from this starting point
    if time.time() < deadline:
        x0 = base_points.flatten()
        res = minimize(objective_flat, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-10})
        if time.time() < deadline:
            opt_points = project_to_triangle(res.x.reshape(-1, 2))
            opt_val = min_triangle_area(opt_points)
            if opt_val > best_val:
                best_val = opt_val
                best_points = opt_points.copy()
    
    # Try more random restarts
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
        
        # Random initial points
        x_init = rng.rand(N, 2)
        # Scale to fit in triangle
        x_init[:, 0] = x_init[:, 0] * 0.8 + 0.1
        x_init[:, 1] = x_init[:, 1] * 0.8 * h + 0.1 * h
        x_init = project_to_triangle(x_init)
        
        x0 = x_init.flatten()
        res = minimize(objective_flat, x0, method='Nelder-Mead',
                       options={'maxiter': 300, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if time.time() > deadline - 5.0:
            break
        
        opt_points = project_to_triangle(res.x.reshape(-1, 2))
        opt_val = min_triangle_area(opt_points)
        if opt_val > best_val:
            best_val = opt_val
            best_points = opt_points.copy()
    
    return best_points

