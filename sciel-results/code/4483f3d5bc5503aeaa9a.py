import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 13

def triangle_area(p1, p2, p3):
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def convex_hull_area(points):
    # Use scipy.spatial.ConvexHull for robustness
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume # In 2D, volume is area
    except Exception:
        return 0.0

def objective(points_flat):
    points = points_flat.reshape(-1, 2)
    # Check for degeneracy
    if np.allclose(points, points[0]):
        return 0.0
    
    areas = []
    for i, j, k in combinations(range(N), 3):
        areas.append(triangle_area(points[i], points[j], points[k]))
    
    min_area = min(areas)
    hull_area = convex_hull_area(points)
    
    if hull_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    initial_points = np.column_stack((x, y))
    
    best_points = initial_points
    best_score = objective(initial_points.flatten())
    
    # Use Nelder-Mead to maximize the objective (minimize negative)
    # We will run optimization for a while, checking time periodically
    # Nelder-Mead is derivative-free and good for this size
    
    def neg_objective(x_flat):
        return -objective(x_flat)
    
    try:
        # Nelder-Mead settings
        options = {
            'maxiter': 1000,
            'xatol': 1e-10,
            'fatol': 1e-10,
            'adaptive': True
        }
        
        result = minimize(neg_objective, initial_points.flatten(), method='Nelder-Mead', options=options)
        
        if result.fun < -best_score: # Since we minimized -obj, result.fun is -max_score
            best_score = -result.fun
            best_points = result.x.reshape(-1, 2)
            
    except Exception as e:
        pass # Keep initial if optimization fails

    # Check time, if we have plenty, we could try other initializations
    # But regular polygon is likely optimal.
    
    return best_points
