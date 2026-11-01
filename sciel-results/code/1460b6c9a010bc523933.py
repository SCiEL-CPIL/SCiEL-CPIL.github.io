import numpy as np
from scipy.spatial import ConvexHull
import time

N = 14

def min_triangle_area_ratio(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
    
    if hull_area <= 1e-12:
        return 0.0
    
    # Compute all triangle areas
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = pts[i], pts[j], pts[k]
                area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    # Start with regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = np.column_stack((np.cos(angles), np.sin(angles)))
    best_pts = pts.copy()
    best_score = min_triangle_area_ratio(best_pts)
    
    # Try local optimization
    from scipy.optimize import minimize
    
    def neg_score(x):
        pts = x.reshape(N, 2)
        return -min_triangle_area_ratio(pts)
    
    x0 = best_pts.flatten()
    
    try:
        result = minimize(neg_score, x0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        
        if time.time() < deadline:
            candidate = result.x.reshape(N, 2)
            cand_score = min_triangle_area_ratio(candidate)
            if cand_score > best_score:
                best_pts = candidate
                best_score = cand_score
    except Exception:
        pass
    
    return best_pts
