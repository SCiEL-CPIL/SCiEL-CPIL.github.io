import numpy as np
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    
    # Initial guess: Regular polygon
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts_init = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Flatten pts to 1D array for optimizer
    x0 = pts_init.flatten()
    
    def calc_score_from_pts(pts):
        # Normalize
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        dists_to_centroid = np.linalg.norm(pts, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        if mean_dist <= 0.01:
            return 0.0
        pts = pts / mean_dist
        
        # Check convexity and calculate score
        # 1. Check for repeated points
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(pts[i] - pts[j]) < 1e-6:
                    return 0.0
        
        # 2. Check strict convexity (cross products same sign, non-zero)
        cross_products = []
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i+1)%n]
            p2 = pts[(i+2)%n]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            cross_products.append(cross)
        
        cross_products = np.array(cross_products)
        if np.any(cross_products == 0) or np.any(np.sign(cross_products) != np.sign(cross_products[0])):
            return 0.0
            
        # 3. Calculate vertex scores
        min_vertex_score = 1.0
        for i in range(n):
            v = pts[i]
            dists = np.linalg.norm(pts - v, axis=1)
            dists = np.delete(dists, i)
            dists = np.sort(dists)
            
            if dists[0] < 0.05:
                return 0.0
                
            diffs = np.diff(dists)
            k = np.argmin(diffs)
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
            score = min(1.0, 1.0 / raw)
            min_vertex_score = min(min_vertex_score, score)
            
        return min_vertex_score

    def objective(x):
        pts = x.reshape(n, 2)
        score = calc_score_from_pts(pts)
        # We want to maximize score, so minimize -score
        # If score is 0, return a large penalty
        if score == 0.0:
            return 100.0
        return -score

    # Optimize
    # Nelder-Mead is derivative-free and works well for small N
    # We limit max iterations to ensure it finishes in time
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    best_pts = result.x.reshape(n, 2)
    
    # Final check and normalization
    final_score = calc_score_from_pts(best_pts)
    
    # If optimization failed or resulted in invalid polygon, fallback to regular
    if final_score < 1e-6:
        return pts_init
        
    # Normalize the result as per task description
    centroid = np.mean(best_pts, axis=0)
    best_pts = best_pts - centroid
    dists_to_centroid = np.linalg.norm(best_pts, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    best_pts = best_pts / mean_dist
    
    return best_pts
