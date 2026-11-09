import numpy as np
import time
import sys

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    
    n = 20
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # Initial radii
    r = np.ones(n)
    
    def get_polygon(r):
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        return np.column_stack((x, y))
    
    def score_func(r):
        # Check time
        if time.time() > deadline - 10.0:
            return 0.0 # or current best
        
        polygon = get_polygon(r)
        
        # Check convexity and distinctness
        # Cross products
        for i in range(n):
            p0 = polygon[i]
            p1 = polygon[(i+1)%n]
            p2 = polygon[(i+2)%n]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            if cross < 1e-6: # Must be strictly convex, CCW
                return 0.0
            if np.linalg.norm(p1-p0) < 0.01:
                return 0.0

        # Calculate min vertex score
        min_score = 1.0
        for i in range(n):
            v = polygon[i]
            dists = np.linalg.norm(polygon - v, axis=1)
            dists = np.delete(dists, i)
            dists = np.sort(dists)
            
            if dists[0] < 0.05:
                return 0.0
            
            diffs_d = np.diff(dists)
            min_diff = np.min(diffs_d)
            k = np.argmin(diffs_d)
            
            D = (dists[k] + dists[k+1]) / 2.0
            if D < 0.1:
                return 0.0
            
            if k == 0:
                sel = dists[0:4]
            elif k == 17:
                sel = dists[15:19]
            else:
                sel = dists[k-1:k+3]
                
            ratios = np.maximum(sel / D, D / sel)
            raw = np.mean(ratios)
            v_score = min(1.0, 1.0 / raw)
            
            if v_score < min_score:
                min_score = v_score
        
        return min_score

    # Optimization
    # Use Nelder-Mead
    # Objective: minimize -score
    def objective(r):
        s = score_func(r)
        return -s

    # Simple manual Nelder-Mead or use scipy if available?
    # Prompt says SciPy is available.
    from scipy.optimize import minimize
    
    # Bounds for r
    bounds = [(0.5, 1.5)] * n
    
    best_r = r.copy()
    best_score = score_func(best_r)
    
    # Run optimization
    # Nelder-Mead doesn't support bounds directly in older scipy, 
    # but we can penalize out of bounds or use L-BFGS-B with gradient?
    # Gradient is hard.
    # Let's use Nelder-Mead and clamp r in get_polygon? No, that breaks differentiability.
    # Better: penalize out of bounds in objective.
    
    def objective_penalized(r):
        if np.any(r < 0.5) or np.any(r > 1.5):
            return 100.0
        return -score_func(r)
        
    try:
        res = minimize(objective_penalized, r, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
        if res.x is not None:
            # Check if better
            s = score_func(res.x)
            if s > best_score:
                best_r = res.x
                best_score = s
    except Exception:
        pass
        
    # Final polish or return
    polygon = get_polygon(best_r)
    
    # Ensure normalization is handled by evaluator, 
    # but we can normalize here to be safe? 
    # "The polygon is normalized: subtract centroid, divide by mean distance"
    # This is done by evaluator.
    
    return polygon
