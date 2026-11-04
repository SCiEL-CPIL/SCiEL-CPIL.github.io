"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from itertools import combinations
import math

def _triangle_area(p1, p2, p3):
    """Area of triangle with vertices p1, p2, p3 (2D points)."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                     (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i, j, k in combinations(range(n), 3):
        a = _triangle_area(points[i], points[j], points[k])
        if a < min_area:
            min_area = a
    return min_area

def _construct_from_params(params):
    """
    Construct 11 points from 13 parameters.
    Params:
    0: x1 (fraction along bottom edge from (0,0) to (1,0))
    1: x2 (fraction along right edge from (1,0) to (0.5, h))
    2: x3 (fraction along left edge from (0,0) to (0.5, h))
    3-12: x,y coordinates for 5 interior points
    """
    h = math.sqrt(3) / 2
    # Fixed vertices
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, h])
    
    # Edge points
    # Bottom edge: (0,0) to (1,0)
    p1 = np.array([params[0], 0.0])
    
    # Right edge: (1,0) to (0.5, h)
    # P = (1,0) + t * ((0.5, h) - (1,0)) = (1 - 0.5t, h*t)
    t2 = params[1]
    p2 = np.array([1.0 - 0.5 * t2, h * t2])
    
    # Left edge: (0,0) to (0.5, h)
    # P = (0,0) + t * ((0.5, h) - (0,0)) = (0.5t, h*t)
    t3 = params[2]
    p3 = np.array([0.5 * t3, h * t3])
    
    # Interior points
    interior_pts = []
    for i in range(5):
        x = params[3 + 2*i]
        y = params[4 + 2*i]
        # Project to triangle to ensure validity
        # Simple projection: clamp y to [0, h], then x to [y/sqrt(3), 1 - y/sqrt(3)]
        if y < 0: y = 0
        if y > h: y = h
        min_x = y / math.sqrt(3)
        max_x = 1 - y / math.sqrt(3)
        if x < min_x: x = min_x
        if x > max_x: x = max_x
        interior_pts.append(np.array([x, y]))
        
    points = [v0, v1, v2, p1, p2, p3] + interior_pts
    return np.array(points)

def _neg_min_area_from_params(params):
    """Negative of minimum triangle area for optimization."""
    points = _construct_from_params(params)
    min_a = _min_triangle_area(points)
    return -min_a

def construct():
    """Construct 11 points in the equilateral triangle maximizing min triangle area."""
    h = math.sqrt(3) / 2
    
    # Bounds for 13 parameters
    # 3 edge fractions in [0, 1]
    # 5 interior x in [0, 1]
    # 5 interior y in [0, h]
    bounds = [
        (0.0, 1.0), # x1
        (0.0, 1.0), # x2
        (0.0, 1.0), # x3
        (0.0, 1.0), # int1 x
        (0.0, h),   # int1 y
        (0.0, 1.0), # int2 x
        (0.0, h),   # int2 y
        (0.0, 1.0), # int3 x
        (0.0, h),   # int3 y
        (0.0, 1.0), # int4 x
        (0.0, h),   # int4 y
        (0.0, 1.0), # int5 x
        (0.0, h),   # int5 y
    ]
    
    # Initial guess: symmetric
    # Edge points at midpoints (0.5)
    # Interior points: center (0.5, h/3), and 4 others roughly symmetric
    init_params = np.array([
        0.5, 0.5, 0.5,
        0.5, h/3,
        0.25, h/4,
        0.75, h/4,
        0.5, h/2,
        0.5, h/6
    ])
    
    # Run Differential Evolution
    # We have 13 variables. 
    # popsize=20, maxiter=100 should be manageable within 120s.
    try:
        res = differential_evolution(
            _neg_min_area_from_params, 
            bounds, 
            init=init_params,
            seed=42, 
            maxiter=100, 
            popsize=20, 
            tol=1e-5,
            polish=True # Use L-BFGS-B for polish
        )
        best_params = res.x
    except Exception:
        # Fallback to initial guess
        best_params = init_params
        
    points = _construct_from_params(best_params)
    return points
# EVOLVE-BLOCK-END
