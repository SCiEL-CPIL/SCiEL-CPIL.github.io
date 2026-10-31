import numpy as np
import time
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer for safety
    
    V1 = np.array([1.0, 0.0])
    V2 = np.array([0.5, np.sqrt(3)/2])
    
    def get_points_from_params(params):
        v = params[0::2]
        w = params[1::2]
        points = v[:, None] * V1 + w[:, None] * V2
        return points

    def min_triangle_area(points):
        n = points.shape[0]
        min_a = float('inf')
        for i in range(n):
            p1 = points[i]
            for j in range(i+1, n):
                p2 = points[j]
                dx1 = p2[0] - p1[0]
                dy1 = p2[1] - p1[1]
                for k in range(j+1, n):
                    p3 = points[k]
                    dx2 = p3[0] - p1[0]
                    dy2 = p3[1] - p1[1]
                    cross = dx1 * dy2 - dy1 * dx2
                    area = 0.5 * abs(cross)
                    if area < min_a:
                        min_a = area
        return min_a

    def objective(params):
        v = params[0::2]
        w = params[1::2]
        if np.any(v < -1e-8) or np.any(w < -1e-8) or np.any(v + w > 1 + 1e-8):
            return 1e6
        points = get_points_from_params(params)
        return -min_triangle_area(points)

    best_score = -float('inf')
    best_params = None
    
    rng = np.random.RandomState(123)
    
    inits = []
    
    # 1. Random Dirichlet inits
    for _ in range(10):
        coords = rng.dirichlet([1, 1, 1], size=N)
        v_comp = coords[:, 1]
        w_comp = coords[:, 2]
        params = np.zeros(2*N)
        params[0::2] = v_comp
        params[1::2] = w_comp
        inits.append(params)
        
    # 2. Lattice Init (10 points from 4x4 grid + 1 centroid)
    lattice_params = np.zeros(2*N)
    idx = 0
    for i in range(4):
        for j in range(4 - i):
            if idx < N:
                lattice_params[2*idx] = i / 4.0
                lattice_params[2*idx + 1] = j / 4.0
                idx += 1
    lattice_params[2*10] = 1/3.0
    lattice_params[2*10 + 1] = 1/3.0
    inits.append(lattice_params)
    
    # 3. Another lattice variant: 5x5 grid, pick 11
    lattice_params2 = np.zeros(2*N)
    idx = 0
    for i in range(5):
        for j in range(5 - i):
            if idx < N:
                lattice_params2[2*idx] = i / 5.0
                lattice_params2[2*idx + 1] = j / 5.0
                idx += 1
    inits.append(lattice_params2)

    for x0 in inits:
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                
        except Exception:
            pass
            
    # Polish
    if best_params is not None and time.time() < deadline - 10:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        coords = rng.dirichlet([1, 1, 1], size=N)
        v_comp = coords[:, 1]
        w_comp = coords[:, 2]
        best_params = np.zeros(2*N)
        best_params[0::2] = v_comp
        best_params[1::2] = w_comp
        
    points = get_points_from_params(best_params)
    return points
