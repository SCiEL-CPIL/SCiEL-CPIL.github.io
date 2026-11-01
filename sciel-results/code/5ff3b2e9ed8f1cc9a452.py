import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def compute_ratio(points):
    """
    Calculate the ratio: min triangle area / convex hull area.
    """
    try:
        # Calculate min triangle area
        # Vectorized calculation
        i, j, k = np.indices((N, N, N))
        mask = (i < j) & (j < k)
        ii = i[mask]
        jj = j[mask]
        kk = k[mask]
        
        pi = points[ii]
        pj = points[jj]
        pk = points[kk]
        
        v1 = pj - pi
        v2 = pk - pi
        areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        min_area = np.min(areas)
        
        # Calculate convex hull area
        # Ensure points are not collinear
        hull = ConvexHull(points)
        hull_area = hull.volume
        
        if hull_area == 0:
            return 0.0
            
        return min_area / hull_area
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = 100.0 # Safety margin before 120s/150s limit
    
    # 1. Initial guess: Regular N-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack((np.cos(angles), np.sin(angles))).flatten()
    
    best_points = x0
    best_score = compute_ratio(x0.reshape(N, 2))
    
    # 2. Local Optimization
    def objective(x):
        if time.time() > deadline:
            return 0.0
        pts = x.reshape(N, 2)
        r = compute_ratio(pts)
        return -r # Minimize negative ratio
        
    try:
        # Nelder-Mead is good for non-smooth functions
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if time.time() < deadline:
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_points = res.x
    except Exception:
        pass
        
    return best_points.reshape(N, 2)
