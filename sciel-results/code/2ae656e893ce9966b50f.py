"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
S3 = 3 ** 0.5
H = S3 / 2

# Precompute triples for vectorized area calculation
TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

def _min_area_and_penalty(params):
    """
    Returns (min_area, penalty).
    min_area is the min triangle area.
    penalty is 0 if all points are inside, >0 otherwise.
    """
    points = params.reshape(N, 2)
    x = points[:, 0]
    y = points[:, 1]
    
    # Penalty calculation
    max_y = H * np.minimum(x, 1 - x)
    # Violations
    v_y = np.maximum(0, -y)
    v_x0 = np.maximum(0, -x)
    v_x1 = np.maximum(0, x - 1)
    v_top = np.maximum(0, y - max_y)
    penalty = np.sum(v_y**2 + v_x0**2 + v_x1**2 + v_top**2)
    
    # Area calculation
    p1 = points[TRIPLES[:, 0]]
    p2 = points[TRIPLES[:, 1]]
    p3 = points[TRIPLES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_a = np.min(areas)
    
    return min_a, penalty

def _obj_de(params):
    min_a, pen = _min_area_and_penalty(params)
    if pen > 1e-12:
        return 100 * pen
    return -min_a

def _obj_slsqp(params):
    min_a, pen = _min_area_and_penalty(params)
    if pen > 1e-12:
        return 100 * pen
    return -min_a

def construct():
    """Construct 11 points in the equilateral triangle to maximize min triangle area."""
    bounds = [(0, 1), (0, H)] * N
    
    # 1. Global Search with Differential Evolution
    try:
        res_de = differential_evolution(
            _obj_de, 
            bounds=bounds, 
            seed=42, 
            maxiter=200, 
            popsize=20, 
            tol=1e-8,
            polish=False
        )
        best_x = res_de.x
    except Exception:
        # Fallback to random points inside
        best_x = np.random.rand(N, 2)
        best_x[:, 1] *= H
        for i in range(N):
            if best_x[i][1] > H * min(best_x[i][0], 1-best_x[i][0]):
                best_x[i][1] = H * min(best_x[i][0], 1-best_x[i][0])

    # 2. Local Refinement with SLSQP
    try:
        res_slsqp = minimize(
            _obj_slsqp, 
            best_x, 
            method='SLSQP', 
            bounds=bounds, 
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        if res_slsqp.fun < _obj_slsqp(best_x):
            best_x = res_slsqp.x
    except Exception:
        pass
        
    # 3. Final Check and Projection
    points = best_x.reshape(N, 2)
    for i in range(N):
        x, y = points[i]
        if y < 0: y = 0
        if x < 0: x = 0
        if x > 1: x = 1
        max_y = H * min(x, 1-x)
        if y > max_y: y = max_y
        points[i] = [x, y]
        
    return points
