"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools
import time

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def point_in_triangle(p, a=V0, b=V1, c=V2, eps=1e-9):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)

def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                                  (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area

def negative_min_area_flat(x):
    points = x.reshape(-1, 2)
    # Penalize points outside triangle
    penalty = 0.0
    for p in points:
        if not point_in_triangle(p):
            penalty += 0.1
    return -min_triangle_area(points) + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial configuration: place points in a grid-like pattern inside the triangle
    rng = np.random.RandomState(42)
    
    def initial_points():
        pts = []
        # Place points at vertices and centroid first
        pts.append(V0 + 0.05 * np.array([0.5, np.sqrt(3)/6]))
        pts.append(V1 + 0.05 * np.array([-0.5, np.sqrt(3)/6]))
        pts.append(V2 + 0.05 * np.array([0, -np.sqrt(3)/3]))
        
        # Add more points in a pattern
        for i in range(8):
            u = rng.random() * 0.9 + 0.05
            v = rng.random() * (1 - u) * 0.9 + 0.05
            p = u * V1 + v * V2
            pts.append(p)
        return np.array(pts)
    
    best_points = None
    best_score = -float('inf')
    
    # Try multiple initial configurations
    for restart in range(10):
        if time.time() > deadline - 30.0:
            break
        
        points = initial_points()
        x0 = points.flatten()
        
        # Nelder-Mead optimization
        result = minimize(
            negative_min_area_flat,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        if time.time() > deadline - 10.0:
            break
        
        optimized_points = result.x.reshape(-1, 2)
        
        # Check if all points are inside triangle
        valid = all(point_in_triangle(p) for p in optimized_points)
        
        if valid:
            score = min_triangle_area(optimized_points)
            if score > best_score:
                best_score = score
                best_points = optimized_points
    
    if best_points is None:
        # Fallback: return initial points
        best_points = initial_points()
    
    return best_points
# EVOLVE-BLOCK-END
