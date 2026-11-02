"""Heilbronn triangles: place 11 points in a unit equilateral triangle to maximize the minimum triangle area."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def _penalty(points):
    """
    Calculate penalty for points outside the triangle.
    Triangle vertices: (0,0), (1,0), (0.5, S3/2)
    """
    penalty = 0.0
    for i in range(N):
        x, y = points[i]
        if y < 0:
            penalty += y ** 2
        if y > S3 * x:
            penalty += (y - S3 * x) ** 2
        if y > S3 * (1 - x):
            penalty += (y - S3 * (1 - x)) ** 2
    return penalty

def _objective(params):
    """
    Objective function to minimize (negative of min triangle area + penalty).
    """
    points = params.reshape(N, 2)
    pen = _penalty(points)
    if pen > 1e-8:
        return 1000.0 + pen
    min_area = _min_area_vec(points)
    return -min_area

def construct():
    """
    Place 11 points in the unit equilateral triangle to maximize the minimum triangle area.
    Uses Differential Evolution followed by Nelder-Mead polishing.
    """
    # Define bounds for Differential Evolution
    bounds = []
    for _ in range(N):
        bounds.append((0, 1))
        bounds.append((0, S3 / 2))
    
    # Run Differential Evolution
    np.random.seed(42)
    
    res_de = differential_evolution(
        _objective,
        bounds=bounds,
        seed=42,
        maxiter=200,
        popsize=20,
        tol=1e-6,
        init='latinhypercube'
    )
    
    best_points = res_de.x.reshape(N, 2)
    best_score = -res_de.fun
    
    # Polishing with Nelder-Mead
    n_polish_restarts = 20
    for i in range(n_polish_restarts):
        if i == 0:
            start_points = best_points
        else:
            # Generate random points inside the triangle
            start_points = np.zeros((N, 2))
            for j in range(N):
                u = np.random.uniform(0, 1)
                v = np.random.uniform(0, 1)
                if u + v > 1:
                    u = 1 - u
                    v = 1 - v
                x = u * 1 + v * 0.5
                y = u * 0 + v * (S3 / 2)
                start_points[j] = [x, y]
        
        res_nm = minimize(
            _objective,
            start_points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if res_nm.fun < 500: 
            score = -res_nm.fun
            if score > best_score:
                best_score = score
                best_points = res_nm.x.reshape(N, 2)
                
    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
