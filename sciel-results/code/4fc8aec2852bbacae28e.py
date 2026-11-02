"""Heilbronn triangles: place 11 points in a unit equilateral triangle to maximize the minimum triangle area."""
import numpy as np
from scipy.optimize import differential_evolution
import math

N = 11
S3 = math.sqrt(3)

def _min_area_vec(points):
    """
    Compute the minimum area of any triangle formed by three of the given points.
    points: (N, 2) array
    """
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _objective(params):
    """
    Objective function to minimize (negative of min triangle area).
    params: (2*N,) array representing u, v coordinates in [0, 1] for each point.
    """
    # Map from unit square (u, v) to triangle
    # x = u
    # y = v * sqrt(3) * min(u, 1-u)
    points = np.zeros((N, 2))
    for i in range(N):
        u = params[2*i]
        v = params[2*i+1]
        x = u
        y = v * S3 * min(x, 1.0 - x)
        points[i] = [x, y]
    
    min_area = _min_area_vec(points)
    return -min_area

def construct():
    """
    Place 11 points in the unit equilateral triangle to maximize the minimum triangle area.
    Uses Differential Evolution global optimization.
    """
    # Bounds for u and v are [0, 1]
    bounds = [(0, 1)] * (2 * N)
    
    # Differential Evolution is a global optimizer
    res = differential_evolution(
        _objective,
        bounds,
        method='de',
        popsize=15,
        maxiter=500,
        tol=1e-6,
        seed=42,
        polish=True
    )
    
    # Convert best parameters back to points
    best_params = res.x
    points = np.zeros((N, 2))
    for i in range(N):
        u = best_params[2*i]
        v = best_params[2*i+1]
        x = u
        y = v * S3 * min(x, 1.0 - x)
        points[i] = [x, y]
        
    return points

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
