import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 13

def _min_triangle_area(points):
    # Vectorized calculation of min triangle area
    # points: (N, 2)
    # Use broadcasting to compute all cross products
    # This might use too much memory for large N, but N=13 is small.
    # 13*12*11/6 = 286 triangles.
    # We can just loop, it's fast enough.
    min_area = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _regular_polygon(n, center=(0.0, 0.0), radius=1.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # 1. Regular Polygon Baseline
    best_points = _regular_polygon(N)
    best_score = _score(best_points)
    
    # 2. Differential Evolution
    # Bounds: [-1, 1] for x and y
    bounds = [(-1, 1)] * (2 * N)
    
    def objective(x):
        if time.time() - start_time > deadline:
            return 0.0
        points = x.reshape(-1, 2)
        return -_score(points)
        
    try:
        # Use DE to find a good candidate
        # popsize=15, maxiter=50 is a reasonable start
        # tol=1e-6
        res_de = differential_evolution(objective, bounds, maxiter=50, popsize=20, tol=1e-6, seed=42, polish=False)
        
        if res_de.fun < -best_score:
            candidate = res_de.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate
    except Exception:
        pass

    # 3. Local Refinement from Best Points and Regular Polygon
    # Try Nelder-Mead from the current best
    if time.time() - start_time < deadline - 10:
        x0 = best_points.flatten()
        try:
            res_nm = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res_nm.fun < -best_score:
                candidate = res_nm.x.reshape(-1, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate
        except Exception:
            pass

    # 4. Try Nelder-Mead from Regular Polygon (in case DE drifted)
    if time.time() - start_time < deadline - 10:
        x0_reg = _regular_polygon(N).flatten()
        try:
            res_nm_reg = minimize(objective, x0_reg, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res_nm_reg.fun < -best_score:
                candidate = res_nm_reg.x.reshape(-1, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate
        except Exception:
            pass
            
    return best_points
