import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def _score(points):
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points, metric='euclidean')
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max < 1e-12:
        return 0.0
    return (d_min / d_max) ** 2

def _neg_score(x):
    points = x.reshape(14, 3)
    # Remove translational invariance
    points = points - np.mean(points, axis=0)
    return -_score(points)

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    timeout = 600
    best_score = -1
    best_points = None
    
    # Initializations
    inits = []
    
    # 1. Random uniform in cube
    for seed in range(20):
        np.random.seed(seed)
        inits.append(np.random.uniform(-1, 1, (14, 3)))
        
    # 2. Random points on sphere
    for seed in range(20, 40):
        np.random.seed(seed)
        pts = np.random.randn(14, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts / norms)
        
    # 3. Grid-like (FCC or BCC subset?) - Just a regular grid might be a decent start
    # 14 points in a 2x2x4 grid?
    grid = []
    for i in range(2):
        for j in range(2):
            for k in range(4):
                grid.append([i-0.5, j-0.5, k-1.5])
    inits.append(np.array(grid))
    
    for init in inits:
        if time.time() - start_time > timeout - 30:
            break
        try:
            res = minimize(_neg_score, init.flatten(), method='Nelder-Mead', 
                           options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 20000, 'adaptive': True})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_points = res.x.reshape(14, 3)
        except Exception:
            continue
            
    # Polish the best result with a tighter run
    if best_points is not None and time.time() - start_time < timeout - 10:
        try:
            res = minimize(_neg_score, best_points.flatten(), method='Nelder-Mead', 
                           options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 50000, 'adaptive': True})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_points = res.x.reshape(14, 3)
        except Exception:
            pass

    if best_points is None:
        np.random.seed(0)
        best_points = np.random.randn(14, 3)
        
    return best_points
