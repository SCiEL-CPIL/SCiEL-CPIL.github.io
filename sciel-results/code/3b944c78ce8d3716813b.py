"""Heilbronn triangles: maximize the minimum area of any triangle formed by 11 points in an equilateral triangle."""
import numpy as np
from scipy.optimize import minimize
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
            # Compute distance to triangle (approximate by checking barycentric coords)
            # For simplicity, use a large penalty
            penalty += 1000.0
    
    if penalty > 0:
        return penalty
    
    area = min_triangle_area(points)
    return -area

def construct():
    """Construct 11 points inside the equilateral triangle to maximize the minimum triangle area."""
    
    # Start with a reasonable initial configuration
    # Use a grid-like arrangement that respects the triangular shape
    # We'll use multiple restarts with different initial configurations
    
    best_points = None
    best_area = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Try several initial configurations
    for restart in range(20):
        # Generate initial points
        if restart == 0:
            # Start with a regular grid pattern
            points = []
            # Use a hexagonal-like packing
            rows = [5, 3, 3]  # Number of points in each row
            y_positions = [0.1, 0.4, 0.7]
            for r, y in enumerate(y_positions):
                n_in_row = rows[r]
                # Width of the triangle at height y
                # The triangle has vertices (0,0), (1,0), (0.5, sqrt(3)/2)
                # At height y, the left boundary is x = y/sqrt(3) * (1/2) ... let's compute properly
                # Actually, the left side goes from (0,0) to (0.5, sqrt(3)/2), so x_left = y/sqrt(3)
                # The right side goes from (1,0) to (0.5, sqrt(3)/2), so x_right = 1 - y/sqrt(3)
                # Wait, let me recalculate. The triangle has:
                # Bottom: (0,0) to (1,0)
                # Left: (0,0) to (0.5, sqrt(3)/2)
                # Right: (1,0) to (0.5, sqrt(3)/2)
                # At height y (0 <= y <= sqrt(3)/2):
                # Left x: interpolate from 0 to 0.5: x_left = (y / (sqrt(3)/2)) * 0.5 = y/sqrt(3)
                # Right x: interpolate from 1 to 0.5: x_right = 1 - (y / (sqrt(3)/2)) * 0.5 = 1 - y/sqrt(3)
                x_left = y / SQRT3
                x_right = 1 - y / SQRT3
                width = x_right - x_left
                if n_in_row > 0:
                    if n_in_row == 1:
                        x_positions = [(x_left + x_right) / 2]
                    else:
                        x_positions = np.linspace(x_left + 0.05 * width, x_right - 0.05 * width, n_in_row)
                    for x in x_positions:
                        points.append([x, y])
            
            points = np.array(points[:N])
        else:
            # Random initial points inside the triangle
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
        
        # Ensure points are inside the triangle with a small margin
        x = points.flatten()
        
        # Use Nelder-Mead optimization
        try:
            result = minimize(
                objective,
                x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            opt_points = result.x.reshape((N, 2))
            
            # Check if all points are inside the triangle
            valid = True
            for i in range(N):
                if not point_in_triangle(opt_points[i], V0, V1, V2, eps=1e-6):
                    valid = False
                    break
            
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
