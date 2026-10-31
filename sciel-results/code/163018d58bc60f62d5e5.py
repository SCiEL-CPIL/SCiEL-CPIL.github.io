import numpy as np
from scipy.optimize import minimize
import time
import random

N = 11
# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])

def get_min_area(points):
    """
    Calculate the minimum area of all triangles formed by 3 points.
    points: (N, 2) array
    """
    # Get all combinations of 3 indices
    # N=11, so 165 triangles
    # To be fast, we pre-compute indices? 
    # For N=11, itertools.combinations is fast enough if called once.
    # But inside the loop, we want to avoid overhead.
    # We can hardcode the indices or generate them once globally.
    
    pass

# Pre-compute triangle indices
from itertools import combinations
TRI_INDICES = np.array(list(combinations(range(N), 3)))

def calc_min_area(points):
    """
    points: (11, 2)
    returns: min area
    """
    # points[TRI_INDICES[:, 0]] gives (165, 2)
    p1 = points[TRI_INDICES[:, 0]]
    p2 = points[TRI_INDICES[:, 1]]
    p3 = points[TRI_INDICES[:, 2]]
    
    # Vectorized area calculation
    # Area = 0.5 * | (x2-x1)(y3-y1) - (y2-y1)(x3-x1) |
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    
    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def is_inside_triangle(points):
    """
    Check if all points are inside or on the boundary of the triangle.
    """
    # Barycentric coordinates method or half-plane method
    # Half-plane:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x
    # 3. y <= -sqrt(3) * (x - 1)
    
    x = points[:, 0]
    y = points[:, 1]
    
    eps = 1e-9
    cond1 = y >= -eps
    cond2 = y <= np.sqrt(3) * x + eps
    cond3 = y <= -np.sqrt(3) * (x - 1) + eps
    
    return np.all(cond1) and np.all(cond2) and np.all(cond3)

def penalty(points):
    """
    Calculate penalty for points outside the triangle.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Distance to edges
    # Edge 1: y=0 -> dist = y
    # Edge 2: y - sqrt(3)x = 0 -> dist = (y - sqrt(3)x) / sqrt(1+3) = (y - sqrt(3)x)/2
    # Edge 3: y + sqrt(3)x - sqrt(3) = 0 -> dist = (y + sqrt(3)x - sqrt(3)) / 2
    
    d1 = np.maximum(0, -y)
    d2 = np.maximum(0, (y - np.sqrt(3)*x) / 2)
    d3 = np.maximum(0, (y + np.sqrt(3)*x - np.sqrt(3)) / 2)
    
    # Sum of squared distances
    return np.sum(d1**2) + np.sum(d2**2) + np.sum(d3**2)

def objective_wrapper(points_flat):
    """
    Objective for scipy.optimize.minimize.
    Minimize -min_area + penalty.
    """
    points = points_flat.reshape(N, 2)
    min_area = calc_min_area(points)
    pen = penalty(points)
    # We want to maximize min_area, so minimize -min_area.
    # If outside, penalty dominates.
    return -min_area + 100.0 * pen

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Simulated Annealing
    rng = np.random.default_rng(42)
    
    # Initial guess: Random points inside
    # Use rejection sampling for initial points
    pts = np.zeros((N, 2))
    for i in range(N):
        while True:
            x = rng.random()
            y = rng.random() * np.sqrt(3)/2
            if y <= np.sqrt(3)*x and y <= -np.sqrt(3)*(x-1):
                pts[i] = [x, y]
                break
    
    curr_pts = pts.copy()
    curr_val = calc_min_area(curr_pts)
    best_pts = curr_pts.copy()
    best_val = curr_val
    
    T = 0.01  # Initial temperature
    T_min = 1e-6
    alpha = 0.999  # Cooling rate
    step_size = 0.05
    
    # Run SA
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Generate new point
        noise = rng.normal(0, step_size, size=(N, 2))
        new_pts = curr_pts + noise
        
        # Check validity
        if is_inside_triangle(new_pts):
            new_val = calc_min_area(new_pts)
            delta = new_val - curr_val
            
            # Metropolis criterion
            if delta > 0 or rng.random() < np.exp(delta / T):
                curr_pts = new_pts
                curr_val = new_val
                
                if curr_val > best_val:
                    best_val = curr_val
                    best_pts = curr_pts.copy()
        else:
            # If outside, reject (or reflect, but rejection is simpler and safe)
            pass
            
        T *= alpha
        # Occasionally adjust step size?
        # If we haven't improved in a while, maybe reduce step size?
        # Simple exponential cooling is usually fine.
        
        # Check time inside loop for long steps (though SA steps are fast)
        if time.time() > deadline:
            break
            
    # 2. Local Refinement using Nelder-Mead
    # Start from best_pts
    x0 = best_pts.flatten()
    
    # Check time
    if time.time() < deadline - 10: # Leave 10s for minimize
        try:
            # Bounds are handled by penalty in objective
            result = minimize(objective_wrapper, x0, method='Nelder-Mead', 
                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if result.success or result.fun < -best_val + 1e-6:
                new_best_pts = result.x.reshape(N, 2)
                # Verify it's actually inside
                if is_inside_triangle(new_best_pts):
                    new_best_val = calc_min_area(new_best_pts)
                    if new_best_val > best_val:
                        best_pts = new_best_pts
                        best_val = new_best_val
        except Exception:
            pass
            
    return best_pts
