import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 13

def score_from_angles(angles):
    # angles: array of shape (N,)
    # Ensure sorted? No, the points are just defined by angles.
    # But ConvexHull will handle the order.
    pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-6:
        return 0.0
        
    min_area = float('inf')
    # Use the vectorized-ish loop
    for i in range(N):
        p1 = pts[i]
        for j in range(i+1, N):
            p2 = pts[j]
            v1 = p2 - p1
            for k in range(j+1, N):
                p3 = pts[k]
                v2 = p3 - p1
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial regular polygon
    init_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    best_score = score_from_angles(init_angles)
    best_angles = init_angles.copy()
    
    rng = np.random.default_rng(42)
    
    # Optimize angles
    def neg_score_angles(x):
        return -score_from_angles(x)
        
    # Try a few restarts
    for restart in range(10):
        if time.time() > deadline - 30: # Leave time for 2D refinement
            break
            
        # Perturb angles
        # Add small noise
        noise = rng.normal(0, 0.1, N)
        x0 = init_angles + noise
        # Sort? No, angles don't need to be sorted for the calculation, 
        # but for Nelder-Mead, the parameter space is R^N.
        
        try:
            res = minimize(neg_score_angles, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < -best_score:
                best_score = -res.fun
                best_angles = res.x
        except Exception:
            pass
            
    # Convert best angles to points
    best_points = np.column_stack((np.cos(best_angles), np.sin(best_angles)))
    
    # 2D Refinement
    # If time allows, refine in 2D
    if time.time() < deadline - 10:
        def neg_score_2d(x):
            pts = x.reshape(-1, 2)
            # Compute score for 2D points
            try:
                hull = ConvexHull(pts)
                hull_area = hull.volume
            except Exception:
                return 0.0
            if hull_area < 1e-6: return 0.0
            
            min_area = float('inf')
            for i in range(N):
                p1 = pts[i]
                for j in range(i+1, N):
                    p2 = pts[j]
                    v1 = p2 - p1
                    for k in range(j+1, N):
                        p3 = pts[k]
                        v2 = p3 - p1
                        cross = v1[0]*v2[1] - v1[1]*v2[0]
                        area = 0.5 * abs(cross)
                        if area < min_area:
                            min_area = area
            return - (min_area / hull_area)
            
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score_2d, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < -best_score:
                best_score = -res.fun
                best_points = res.x.reshape(-1, 2)
        except Exception:
            pass
            
    return best_points
