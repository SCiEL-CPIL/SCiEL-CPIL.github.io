import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import time

N = 13

def compute_min_triangle_area(points):
    n = points.shape[0]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def compute_hull_area(points):
    try:
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def score(points):
    hull_area = compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri_area = compute_min_triangle_area(points)
    return min_tri_area / hull_area

def construct():
    # Bounds for each coordinate
    bounds = [(-1, 1)] * (2 * N)
    
    def objective(x):
        points = x.reshape(N, 2)
        s = score(points)
        if s == 0.0:
            return 1e6 # Penalty
        return -s

    # Use Differential Evolution
    # maxiter=100, popsize=15 is a good start
    # We need to be careful with time.
    # DE can be slow.
    # Let's try a smaller population and fewer iterations first, then refine.
    
    # To ensure we finish in 120s, we might need to limit the DE run.
    # Or use a faster global optimizer.
    
    # Alternative: Use the result of DE as a start for Nelder-Mead?
    
    # Let's run DE with a timeout wrapper? No, just set limits.
    # For N=13, 26 variables.
    # DE might be too slow.
    
    # Let's try a hybrid: 
    # 1. Generate random points.
    # 2. Run Nelder-Mead from best random point.
    # This is "Multi-start Nelder-Mead".
    
    np.random.seed(42)
    best_x = None
    best_score = -1.0
    
    # Start 1: Regular Polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    x0 = points.flatten()
    
    from scipy.optimize import minimize
    
    def obj(x):
        pts = x.reshape(N, 2)
        s = score(pts)
        if s == 0.0: return 1e6
        return -s

    # Run Nelder-Mead from Regular Polygon
    res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
    if res.fun < 1e5:
        best_x = res.x
        best_score = -res.fun

    # Run Nelder-Mead from Random Points
    for i in range(20):
        # Random points in unit circle
        r = np.random.rand(N)
        theta = np.random.rand(N) * 2 * np.pi
        pts = np.column_stack([np.sqrt(r) * np.cos(theta), np.sqrt(r) * np.sin(theta)])
        x0_rand = pts.flatten()
        
        res_rand = minimize(obj, x0_rand, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
        if res_rand.fun < 1e5:
            s_rand = -res_rand.fun
            if s_rand > best_score:
                best_score = s_rand
                best_x = res_rand.x

    return best_x.reshape(N, 2)
