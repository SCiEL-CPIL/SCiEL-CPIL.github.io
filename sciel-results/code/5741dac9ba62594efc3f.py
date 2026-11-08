"""Heilbronn triangles: maximize the minimum area of any triangle formed by 11 points in an equilateral triangle."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11
SQRT3 = np.sqrt(3)

# Vertices of the unit equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, SQRT3 / 2.0])

def point_in_triangle(p, v0, v1, v2, eps=1e-10):
    """Check if point p is inside or on the boundary of triangle v0, v1, v2."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, v0, v1)
    d2 = sign(p, v1, v2)
    d3 = sign(p, v2, v0)
    
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    
    return not (has_neg and has_pos)

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |cross product|
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def objective(x):
    """Objective to minimize: negative of the minimum triangle area."""
    points = x.reshape((N, 2))
    
    # Penalty for points outside the triangle
    penalty = 0.0
    for i in range(N):
        p = points[i]
        if not point_in_triangle(p, V0, V1, V2, eps=-1e-6):
            penalty += 1000.0
    
    if penalty > 0:
        return penalty
    
    area = min_triangle_area(points)
    return -area

def construct():
    """Construct 11 points inside the equilateral triangle to maximize the minimum triangle area."""
    
    best_points = None
    best_area = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Use differential evolution for global search
    # Define bounds for each coordinate
    # x ranges from 0 to 1, y ranges from 0 to sqrt(3)/2
    # But we need to ensure points are inside the triangle
    # We'll use a transformation or just check validity
    
    bounds = [(0.0, 1.0)] * N + [(0.0, SQRT3/2)] * N
    
    def de_objective(x):
        points = np.array([[x[2*i], x[2*i+1]] for i in range(N)])
        # Check if all points are inside the triangle
        for i in range(N):
            if not point_in_triangle(points[i], V0, V1, V2, eps=1e-8):
                return 1000.0
        return -min_triangle_area(points)
    
    try:
        de_result = differential_evolution(
            de_objective,
            bounds,
            seed=42,
            maxiter=100,
            tol=1e-6,
            popsize=20,
            init='latinhypercube'
        )
        
        if de_result.fun < 100:  # Valid solution
            de_points = np.array([[de_result.x[2*i], de_result.x[2*i+1]] for i in range(N)])
            valid = all(point_in_triangle(de_points[i], V0, V1, V2, eps=1e-6) for i in range(N))
            if valid:
                area = min_triangle_area(de_points)
                if area > best_area:
                    best_area = area
                    best_points = de_points.copy()
    except Exception:
        pass
    
    # Phase 2: Local refinement using Nelder-Mead from multiple starting points
    starting_points = []
    
    # Add the DE result if we have one
    if best_points is not None:
        starting_points.append(best_points.flatten())
    
    # Add several random valid configurations
    for restart in range(15):
        points = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    x = u + 0.5 * v
                    y = (SQRT3 / 2) * v
                    if point_in_triangle(np.array([x, y]), V0, V1, V2, eps=1e-8):
                        points[i] = [x, y]
                        break
        starting_points.append(points.flatten())
    
    # Add a structured initial configuration
    # Use a hexagonal-like pattern
    points_struct = np.zeros((N, 2))
    # Place points in rows
    row_data = [
        (0.15, 0.15, 3),  # (y, x_center, n_points)
        (0.35, 0.5, 4),
        (0.55, 0.5, 4),
    ]
    idx = 0
    for y, x_center, n_pts in row_data:
        # Width of triangle at height y
        x_left = y / SQRT3
        x_right = 1 - y / SQRT3
        width = x_right - x_left
        if n_pts == 1:
            x_pos = [x_center]
        else:
            x_pos = np.linspace(x_left + 0.1*width, x_right - 0.1*width, n_pts)
        for x in x_pos:
            if idx < N:
                points_struct[idx] = [x, y]
                idx += 1
    
    if all(point_in_triangle(points_struct[i], V0, V1, V2, eps=1e-8) for i in range(N)):
        starting_points.append(points_struct.flatten())
    
    # Refine each starting point
    for x0 in starting_points:
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            opt_points = result.x.reshape((N, 2))
            
            # Check if all points are inside the triangle
            valid = all(point_in_triangle(opt_points[i], V0, V1, V2, eps=1e-6) for i in range(N))
            
            if valid:
                area = min_triangle_area(opt_points)
                if area > best_area:
                    best_area = area
                    best_points = opt_points.copy()
        except Exception:
            continue
    
    if best_points is None:
        # Fallback: return a simple configuration
        points = []
        for i in range(N):
            u = (i + 1) / (N + 1)
            v = 0.3
            if u + v < 1:
                points.append([u + 0.5 * v, (SQRT3 / 2) * v])
            else:
                points.append([0.5, SQRT3 / 6])
        best_points = np.array(points)
    
    return best_points
