"""Heilbronn triangles: place 11 points in a unit equilateral triangle to maximize the minimum triangle area."""
import numpy as np
from scipy.optimize import minimize
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
    Objective function to minimize (negative of min triangle area + penalty).
    """
    points = params.reshape(N, 2)
    
    # Check constraints
    # Triangle vertices: (0,0), (1,0), (0.5, S3/2)
    # y >= 0
    # y <= S3 * x
    # y <= S3 * (1 - x)
    
    penalty = 0.0
    for i in range(N):
        x, y = points[i]
        if y < 0:
            penalty += (0 - y) ** 2
        if y > S3 * x:
            penalty += (y - S3 * x) ** 2
        if y > S3 * (1 - x):
            penalty += (y - S3 * (1 - x)) ** 2
            
    if penalty > 1e-8:
        return 1000.0 + penalty

    min_area = _min_area_vec(points)
    return -min_area

def construct():
    """
    Place 11 points in the unit equilateral triangle to maximize the minimum triangle area.
    Uses Differential Evolution for global optimization.
    """
    
    # Define bounds for Differential Evolution
    # x in [0, 1], y in [0, S3/2]
    bounds = [(0, 1), (0, S3/2)] * N
    
    # Objective function for DE
    # DE minimizes, so we minimize negative min area + penalty
    def de_objective(params):
        points = params.reshape(N, 2)
        penalty = 0.0
        for i in range(N):
            x, y = points[i]
            if y < 0:
                penalty += (0 - y) ** 2
            if y > S3 * x:
                penalty += (y - S3 * x) ** 2
            if y > S3 * (1 - x):
                penalty += (y - S3 * (1 - x)) ** 2
                
        if penalty > 1e-8:
            return 1000.0 + penalty
        
        min_area = _min_area_vec(points)
        return -min_area

    try:
        # Run Differential Evolution
        # popsize=25, maxiter=300 should be fast enough for N=11
        # polish=True uses Nelder-Mead on the best result found
        result = minimize(
            de_objective,
            np.zeros(2 * N), # x0 is ignored for DE
            method='differential_evolution',
            bounds=bounds,
            maxiter=300,
            popsize=25,
            tol=1e-6,
            seed=42,
            polish=True
        )
        
        # Check if result is valid
        if result.fun < 500:
            best_points = result.x.reshape(N, 2)
            return best_points
            
    except Exception:
        pass
        
    # Fallback: Multi-start Nelder-Mead
    best_points = None
    best_score = -float('inf')
    np.random.seed(42)
    
    for _ in range(20):
        points = np.zeros((N, 2))
        for i in range(N):
            x = np.random.uniform(0, 1)
            y_max = S3 * min(x, 1 - x)
            y = np.random.uniform(0, y_max)
            points[i] = [x, y]
        
        res = minimize(
            _objective,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if res.fun < 500: 
            score = -res.fun
            if score > best_score:
                best_score = score
                best_points = res.x.reshape(N, 2)
                
    if best_points is None:
        return np.full((N, 2), [0.5, S3/6])
        
    return best_points
