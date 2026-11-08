"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def score(points):
    # points: (N, 2)
    # Calculate convex hull area
    # Using scipy.spatial.ConvexHull
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-10:
            return 0.0
    except Exception:
        # If hull fails (e.g., collinear points), return 0
        return 0.0
    
    # Calculate min triangle area
    # Number of triangles is C(N, 3)
    # For N=13, C(13,3) = 286. This is fast.
    min_area = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
                area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                if area < min_area:
                    min_area = area
                    
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    init_points = np.column_stack((x, y))
    
    current_best = init_points.copy()
    current_score = score(current_best)
    
    # Local optimization using Nelder-Mead
    # We want to maximize score, so minimize -score
    # Nelder-Mead is derivative-free and robust for small dimensions (2*N = 26 vars)
    
    def neg_score(p):
        return -score(p)
    
    # We might try a few restarts or just one long run
    # Given N=13, 26 dimensions, Nelder-Mead should be okay within time limit.
    
    try:
        result = minimize(neg_score, current_best, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'fatol': 1e-8, 'xatol': 1e-8})
        
        if time.time() < deadline:
            # Check if result is better
            res_score = -result.fun
            if res_score > current_score:
                current_best = result.x
                current_score = res_score
    except Exception:
        pass
        
    # If time remains, we could try another method or more iterations, 
    # but Nelder-Mead with 2000 iterations on 26D is likely fast enough.
    
    return current_best
# EVOLVE-BLOCK-END
