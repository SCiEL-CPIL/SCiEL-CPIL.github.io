"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

N = 11

# EVOLVE-BLOCK-START
def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    return 0.5 * abs(np.cross(p2 - p1, p3 - p1))

def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the unit equilateral triangle."""
    x, y = p
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Check barycentric coordinates
    h = np.sqrt(3) / 2
    # Check y >= 0
    if y < -eps:
        return False
    # Check y <= h
    if y > h + eps:
        return False
    # Check left edge: y <= sqrt(3)*x
    if y > np.sqrt(3) * x + eps:
        return False
    # Check right edge: y <= -sqrt(3)*(x-1) = sqrt(3)*(1-x)
    if y > np.sqrt(3) * (1 - x) + eps:
        return False
    return True

def _maximize_min_area(points):
    """Objective to maximize: negative of min triangle area (for minimization)."""
    return -_min_triangle_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    h = np.sqrt(3) / 2
    # Triangle vertices
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, h])
    
    # Initial template: use a symmetric arrangement
    # For 11 points, we can try a grid-like or symmetric pattern
    # Let's start with points that are well-distributed
    
    # Strategy: Use a parameterized lattice with symmetry
    # The equilateral triangle has 3-fold symmetry. We can exploit this.
    # Place points in orbits under the symmetry group.
    
    # For 11 points: 1 center + orbits of 3 or 6
    # Option: 1 center, 1 orbit of 3 (near vertices), 1 orbit of 3 (mid-edges), 1 orbit of 4? 
    # Actually, let's just do a direct optimization with good initialization.
    
    # Good initial guess: use a known configuration or random restarts
    best_points = None
    best_val = -float('inf')
    
    # Try multiple random restarts with local optimization
    np.random.seed(42)
    
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate random initial points inside the triangle
        points = []
        for _ in range(N):
            # Rejection sampling
            while True:
                u = np.random.random()
                v = np.random.random()
                if u + v <= 1:
                    p = u * v1 + v * v2
                    if _is_inside_triangle(p):
                        points.append(p)
                        break
        points = np.array(points)
        
        # Local optimization using Nelder-Mead
        x0 = points.flatten()
        
        def objective(x):
            pts = x.reshape(N, 2)
            # Check if all points are inside the triangle
            for p in pts:
                if not _is_inside_triangle(p, eps=1e-6):
                    return 1e6  # Penalty
            return _maximize_min_area(pts)
        
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if result.success or result.fun < 1e5:
            pts = result.x.reshape(N, 2)
            # Verify all points are inside
            valid = all(_is_inside_triangle(p) for p in pts)
            if valid:
                val = _min_triangle_area(pts)
                if val > best_val:
                    best_val = val
                    best_points = pts.copy()
    
    # If we found a good solution, try to polish it
    if best_points is not None and time.time() < deadline - 10:
        x0 = best_points.flatten()
        
        def objective(x):
            pts = x.reshape(N, 2)
            for p in pts:
                if not _is_inside_triangle(p, eps=1e-6):
                    return 1e6
            return _maximize_min_area(pts)
        
        # Use BFGS for finer polishing
        result = minimize(objective, x0, method='BFGS', 
                         options={'maxiter': 1000, 'gtol': 1e-10})
        
        if result.fun < 1e5:
            pts = result.x.reshape(N, 2)
            valid = all(_is_inside_triangle(p) for p in pts)
            if valid:
                val = _min_triangle_area(pts)
                if val > best_val:
                    best_val = val
                    best_points = pts.copy()
    
    # Final check: if we didn't find anything, return a default valid configuration
    if best_points is None:
        # Fallback: simple grid-like arrangement
        best_points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, h],
            [0.25, h/2],
            [0.75, h/2],
            [0.5, h/4],
            [0.5, 3*h/4],
            [0.125, h/4],
            [0.875, h/4],
            [0.375, 3*h/4],
            [0.625, 3*h/4],
        ])
    
    return best_points
# EVOLVE-BLOCK-END
