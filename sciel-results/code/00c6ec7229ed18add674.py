"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _negative_min_area(x):
    """Objective: minimize negative of min triangle area."""
    points = x.reshape(N, 2)
    return -_min_triangle_area(points)

def _inside_triangle(p, eps=1e-6):
    """Check if point p is inside the equilateral triangle (0,0),(1,0),(0.5,sqrt(3)/2)."""
    x, y = p
    # Triangle: y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
    if y < -eps or x < -eps or x > 1+eps:
        return False
    if y > sqrt3 * x + eps:
        return False
    if y > sqrt3 * (1 - x) + eps:
        return False
    return True

def _project_to_triangle(points):
    """Project points to be inside the triangle (simple clipping)."""
    for i in range(N):
        x, y = points[i]
        # Clip to triangle
        if y < 0:
            y = 0
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        if y > sqrt3 * x:
            # Project onto line y = sqrt(3)*x
            # The line from (0,0) to (0.5, sqrt(3)/2)
            # Actually the left edge is from (0,0) to (0.5, sqrt(3)/2), slope = sqrt(3)
            # The right edge is from (1,0) to (0.5, sqrt(3)/2), slope = -sqrt(3)
            # For a point above the left edge, project
            # Simple approach: scale down
            if x > 0:
                y = sqrt3 * x
            else:
                y = 0
        if y > sqrt3 * (1 - x):
            if x < 1:
                y = sqrt3 * (1 - x)
            else:
                y = 0
        points[i] = [x, y]
    return points

def _triangular_lattice_init():
    """Generate a triangular lattice initial configuration."""
    points = []
    # Include vertices
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, sqrt3/2])
    # Add points in a triangular grid
    # Row 0 (bottom): already have 2 vertices, add 1 in middle
    points.append([0.5, 0.0])
    # Row 1: y = sqrt(3)/6, x from 1/6 to 5/6
    y1 = sqrt3/6
    points.append([1/6, y1])
    points.append([1/2, y1])
    points.append([5/6, y1])
    # Row 2: y = sqrt(3)/3, x from 1/3 to 2/3
    y2 = sqrt3/3
    points.append([1/3, y2])
    points.append([2/3, y2])
    # Row 3: y = sqrt(3)/2 - small, but we already have top vertex
    # We have 3 + 1 + 3 + 2 = 9 points, need 2 more
    # Add points near the middle
    points.append([0.4, sqrt3/4])
    points.append([0.6, sqrt3/4])
    
    # We have 11 points
    return np.array(points[:N])

def _random_init(seed):
    """Generate a random initial configuration."""
    rng = np.random.RandomState(seed)
    points = []
    for _ in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v < 1:
                x = u + 0.5 * v
                y = (sqrt3 / 2) * v
                points.append([x, y])
                break
    return np.array(points)

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin
    
    best_points = None
    best_area = -float('inf')
    
    # Initial configurations
    inits = []
    
    # Triangular lattice
    try:
        inits.append(_triangular_lattice_init())
    except:
        pass
    
    # Several random inits
    for seed in range(5):
        try:
            inits.append(_random_init(seed))
        except:
            pass
    
    # A more structured init: vertices + centroid + others
    try:
        p = []
        p.append([0.0, 0.0])
        p.append([1.0, 0.0])
        p.append([0.5, sqrt3/2])
        p.append([0.5, sqrt3/6])  # centroid
        # Add points around
        for i in range(7):
            angle = 2 * np.pi * i / 7
            r = 0.3
            x = 0.5 + r * np.cos(angle)
            y = sqrt3/6 + r * np.sin(angle)
            if _inside_triangle([x, y]):
                p.append([x, y])
            else:
                # Adjust
                p.append([0.5 + 0.2*np.cos(angle), sqrt3/6 + 0.2*np.sin(angle)])
        # Ensure we have exactly N points
        while len(p) < N:
            p.append([0.5, sqrt3/6])
        inits.append(np.array(p[:N]))
    except:
        pass
    
    for init in inits:
        if time.time() - start_time > deadline - 10:
            break
        
        try:
            # Ensure init is valid
            init = _project_to_triangle(init.copy())
            
            # Use Nelder-Mead for derivative-free optimization
            res = minimize(
                _negative_min_area,
                init.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'fatol': 1e-8, 'xatol': 1e-8}
            )
            
            if time.time() - start_time > deadline - 5:
                break
            
            # Project result back to triangle
            result_points = _project_to_triangle(res.x.reshape(N, 2).copy())
            
            # Check validity
            valid = True
            for pt in result_points:
                if not _inside_triangle(pt, eps=1e-4):
                    valid = False
                    break
            
            if valid:
                area = _min_triangle_area(result_points)
                if area > best_area:
                    best_area = area
                    best_points = result_points.copy()
                    
        except Exception as e:
            pass
    
    # If no valid solution found, return a simple valid configuration
    if best_points is None:
        # Fallback: triangular lattice
        best_points = _triangular_lattice_init()
        best_points = _project_to_triangle(best_points.copy())
    
    return best_points

sqrt3 = 3 ** 0.5

# EVOLVE-BLOCK-END
