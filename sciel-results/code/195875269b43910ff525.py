"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    h = np.sqrt(3) / 2
    eps = 1e-10
    
    # Constraints:
    # y >= 0
    # y <= h (max height)
    # y <= sqrt(3) * x  (left side, slope = h/0.5 = sqrt(3))
    # y <= sqrt(3) * (1 - x)  (right side)
    
    y = max(eps, y)
    y = min(y, h - eps)
    
    # From y <= sqrt(3)*x => x >= y/sqrt(3)
    x_min_left = y / np.sqrt(3)
    # From y <= sqrt(3)*(1-x) => x <= 1 - y/sqrt(3)
    x_max_right = 1 - y / np.sqrt(3)
    
    x = max(x_min_left + eps, x)
    x = min(x_max_right - eps, x)
    
    return np.array([x, y])

def _evaluate(points):
    """Evaluate the minimum triangle area for a set of points."""
    return _min_triangle_area(points)

def _objective(x_flat, points_list):
    """Objective function for optimization (to minimize negative min area)."""
    points = np.array(points_list).reshape(-1, 2)
    # Clamp all points
    points = np.array([_clamp_to_triangle(p) for p in points])
    min_a = _min_triangle_area(points)
    return -min_a

def construct():
    """
    Place 11 points on or inside a unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    """
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    h = np.sqrt(3) / 2
    
    # Initial good configuration: vertices, midpoints, and center
    # This is a reasonable starting point
    initial_points = [
        [0.0, 0.0],      # vertex 1
        [1.0, 0.0],      # vertex 2
        [0.5, h],         # vertex 3
        [0.25, h/4],      # midpoint of left side
        [0.75, h/4],      # midpoint of right side
        [0.5, h/2],       # midpoint of top... wait, let me think
        [0.5, h/3],       # center
        [0.333, h/3],     
        [0.667, h/3],     
        [0.167, h/6],     
        [0.833, h/6],     
    ]
    
    # Better initial: use a known good configuration
    # Let's start with vertices + centroid + some symmetric points
    initial_points = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],       # centroid
        [0.25, h/6],
        [0.75, h/6],
        [0.5, h/2],
        [0.125, h/12],
        [0.875, h/12],
        [0.375, h/4],
        [0.625, h/4],
    ]
    
    best_points = np.array(initial_points)
    best_score = _evaluate(best_points)
    
    # Local optimization using scipy
    def _local_optimize(points, max_iter=1000):
        """Perform local optimization starting from given points."""
        x0 = np.array(points).flatten()
        
        # Use Nelder-Mead for derivative-free optimization
        result = minimize(
            _objective,
            x0,
            args=(points,),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        opt_points = np.array(result.x).reshape(-1, 2)
        opt_points = np.array([_clamp_to_triangle(p) for p in opt_points])
        return opt_points, _evaluate(opt_points)
    
    # Run local optimization
    if time.time() < deadline:
        opt_points, opt_score = _local_optimize(best_points)
        if opt_score > best_score:
            best_points = opt_points
            best_score = opt_score
    
    # Try multiple random restarts if time permits
    np.random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 30:
            break
        
        # Generate random points inside the triangle
        points = []
        for _ in range(N):
            u = np.random.random()
            v = np.random.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = u + 0.5 * v
            y = h * v
            points.append([x, y])
        
        # Also try some structured random configurations
        if restart % 3 == 0:
            # Symmetric configuration with random perturbations
            base = [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, h],
                [0.5, h/3 + 0.02 * np.random.randn()],
                [0.25 + 0.02 * np.random.randn(), h/6 + 0.01 * np.random.randn()],
                [0.75 + 0.02 * np.random.randn(), h/6 + 0.01 * np.random.randn()],
                [0.5 + 0.02 * np.random.randn(), h/2 + 0.01 * np.random.randn()],
                [0.125 + 0.02 * np.random.randn(), h/12 + 0.01 * np.random.randn()],
                [0.875 + 0.02 * np.random.randn(), h/12 + 0.01 * np.random.randn()],
                [0.375 + 0.02 * np.random.randn(), h/4 + 0.01 * np.random.randn()],
                [0.625 + 0.02 * np.random.randn(), h/4 + 0.01 * np.random.randn()],
            ]
            points = base
        
        # Clamp all points
        points = [_clamp_to_triangle(p) for p in points]
        current_score = _evaluate(points)
        
        if current_score > best_score:
            best_points = np.array(points)
            best_score = current_score
        
        # Local optimization from this starting point
        if time.time() < deadline - 10:
            opt_points, opt_score = _local_optimize(points, max_iter=500)
            if opt_score > best_score:
                best_points = opt_points
                best_score = opt_score
    
    # Final check: ensure all points are inside the triangle
    best_points = np.array([_clamp_to_triangle(p) for p in best_points])
    
    return best_points

# EVOLVE-BLOCK-END
