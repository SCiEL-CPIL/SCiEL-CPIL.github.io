"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy: Direct gradient-based optimization using L-BFGS-B on a smooth 
log-sum-exp approximation of the minimum triangle area. 
This avoids the noise of stochastic search and exploits the differentiability 
of the soft-min surrogate to find precise local optima. Multiple diverse 
initializations are used to cover the landscape.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _soft_min_objective(x_flat, temperature=1.0):
    """
    Smoothed objective: - (soft min of triangle areas) - penalty for being outside.
    Uses log-sum-exp approximation for the minimum to make it differentiable.
    """
    pts = x_flat.reshape(N, 2)
    
    # Penalty for being outside the triangle
    penalty = 0.0
    for p in pts:
        x, y = p
        # Distance to boundaries
        d1 = y # bottom
        d2 = np.sqrt(3) * x - y # left
        d3 = np.sqrt(3) * (1 - x) - y # right
        d4 = x # left vertical (not strictly needed if d2 handles it, but safe)
        d5 = 1 - x # right vertical
        
        # Sum of squared violations
        if d1 < 0: penalty += d1**2
        if d2 < 0: penalty += d2**2
        if d3 < 0: penalty += d3**2
        if d4 < 0: penalty += d4**2
        if d5 < 0: penalty += d5**2
        
    # Compute all triangle areas
    areas = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                areas.append(area)
                
    areas = np.array(areas)
    
    # Soft minimum approximation: min(a) ~ (1/b) * log(sum(exp(-b*a)))
    # We want to maximize the min, so we minimize -min.
    # Let's use: f = - (1/b) * log( sum( exp(-b * area) ) )
    # As b -> inf, this approaches -min(area).
    b = 1.0 / temperature
    # Clip areas to avoid overflow in exp
    areas_clipped = np.clip(areas, 1e-6, None)
    exp_terms = np.exp(-b * areas_clipped)
    soft_min_val = (1.0 / b) * np.log(np.sum(exp_terms))
    
    # We want to maximize soft_min_val, so objective is -soft_min_val
    obj = -soft_min_val
    # Add penalty for being outside
    obj += 100.0 * penalty
    
    return obj

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _initial_guess(rng):
    """Generate an initial set of 11 points."""
    pts = []
    for _ in range(N):
        pts.append(_random_point_in_triangle(rng))
    return np.array(pts)

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses L-BFGS-B on a smooth soft-min objective with multiple restarts.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -float('inf')
    
    # List of initial configurations to try
    initial_configs = []
    
    # 1. Random starts
    for _ in range(5):
        initial_configs.append(_initial_guess(rng))
        
    # 2. Structured start: Corners and center
    corners = np.array([[0,0], [1,0], [0.5, S3H]])
    center = np.array([[0.5, S3H/3]])
    # Add some midpoints
    midpoints = np.array([
        [0.25, S3H/4], [0.75, S3H/4], [0.5, S3H/2],
        [0.5, S3H/6], [0.25, S3H/2], [0.75, S3H/2]
    ])
    structured = np.vstack([corners, center, midpoints])
    initial_configs.append(structured)
    
    # 3. Grid-like start
    grid_pts = []
    for i in range(4):
        for j in range(4):
            if i+j < 4:
                x = 0.2 + 0.3 * i
                y = 0.2 * S3H * j
                # Rough check
                if y < np.sqrt(3)*x and y < np.sqrt(3)*(1-x):
                    grid_pts.append([x, y])
    if len(grid_pts) >= 11:
        initial_configs.append(np.array(grid_pts[:11]))
    else:
        # Fallback if grid was too sparse
        initial_configs.append(_initial_guess(rng))

    # Optimization loop
    for init_pts in initial_configs:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Project to ensure feasibility for the optimizer
        init_pts = _project_to_triangle(init_pts)
        
        # Use a moderately high temperature for the soft-min to ensure smoothness
        # but not so high that it loses fidelity to the true min.
        # b = 1/T. T=0.01 -> b=100.
        temperature = 0.01
        
        try:
            result = minimize(
                lambda x: _soft_min_objective(x, temperature=temperature),
                init_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 200, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            polished_pts = result.x.reshape(N, 2)
            polished_pts = _project_to_triangle(polished_pts)
            
            if _check_inside(polished_pts):
                area = _min_triangle_area(polished_pts)
                if area > best_score:
                    best_score = area
                    best_pts = polished_pts.copy()
                    
                    # If we found a very good solution, we might try to refine it further
                    # with a lower temperature (sharper min)
                    if best_score > 0.005: # Threshold for "good"
                         if time.time() - start_time > DEADLINE - 20:
                             break
                         temp_fine = 0.001
                         try:
                             result_fine = minimize(
                                 lambda x: _soft_min_objective(x, temperature=temp_fine),
                                 polished_pts.flatten(),
                                 method='L-BFGS-B',
                                 options={'maxiter': 50, 'maxfun': 100, 'ftol': 1e-10, 'gtol': 1e-10}
                             )
                             fine_pts = result_fine.x.reshape(N, 2)
                             fine_pts = _project_to_triangle(fine_pts)
                             if _check_inside(fine_pts):
                                 fine_area = _min_triangle_area(fine_pts)
                                 if fine_area > best_score:
                                     best_score = fine_area
                                     best_pts = fine_pts.copy()
                         except Exception:
                             pass
                            
        except Exception:
            pass

    if best_pts is None:
        # Fallback
        best_pts = _initial_guess(rng)
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts
