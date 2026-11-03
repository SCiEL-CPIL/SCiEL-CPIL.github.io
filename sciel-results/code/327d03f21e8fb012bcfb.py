import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def is_inside_triangle(p):
    # p is (2,)
    # Cross product z-component of (B-A) x (P-A)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(V0, V1, p)
    d2 = cross(V1, V2, p)
    d3 = cross(V2, V0, p)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)

def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                # Area = 0.5 * | (pj - pi) x (pk - pi) |
                area = 0.5 * abs((pj[0] - pi[0]) * (pk[1] - pi[1]) - (pk[0] - pi[0]) * (pj[1] - pi[1]))
                if area < min_area:
                    min_area = area
    return min_area

def objective(flat_points):
    points = flat_points.reshape(N, 2)
    
    # Check if all points are inside
    for p in points:
        if not is_inside_triangle(p):
            return 1e6 # Large penalty
    
    area = min_triangle_area(points)
    return -area # Maximize area -> Minimize negative area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Bounds for x and y
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0, 1), (0, np.sqrt(3)/2)] * N
    
    # Differential Evolution
    # We use a relatively small popsize and maxiter to fit in time, 
    # relying on the global nature of DE.
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=0.01, 
            seed=42,
            polish=False # We will polish manually to control time
        )
    except Exception:
        # Fallback to random if DE fails or times out unexpectedly
        res = None

    best_points = None
    best_score = -1e6
    
    if res is not None and res.success:
        best_points = res.x.reshape(N, 2)
        best_score = -res.fun # res.fun is -area, so score is area
    else:
        # If DE didn't finish or failed, try a few random restarts with local optimization
        # Or just return random if time is tight
        pass

    # If DE didn't provide a good result or we have time left, try to polish
    if time.time() < deadline - 10:
        if best_points is None:
            # Initialize with a random valid point
            # Simple rejection sampling for initial guess
            while True:
                p = np.random.rand(N, 2)
                p[:, 0] *= 1.0
                p[:, 1] *= np.sqrt(3)/2
                # Check validity
                valid = all(is_inside_triangle(pt) for pt in p)
                if valid:
                    best_points = p
                    break
        
        # Polish with Nelder-Mead
        try:
            res_nm = minimize(
                objective, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res_nm.fun < objective(best_points.flatten()):
                best_points = res_nm.x.reshape(N, 2)
                best_score = -res_nm.fun
        except Exception:
            pass

    # Final check: if best_points is still None or invalid, generate a valid random set
    if best_points is None or not all(is_inside_triangle(p) for p in best_points):
        while True:
            p = np.random.rand(N, 2)
            p[:, 0] *= 1.0
            p[:, 1] *= np.sqrt(3)/2
            if all(is_inside_triangle(pt) for pt in p):
                best_points = p
                break

    return best_points
