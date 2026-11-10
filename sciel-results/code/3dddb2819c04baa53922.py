import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

def calculate_radius(centers):
    if centers.shape != (N, 3):
        return -np.inf
    if not np.all(np.isfinite(centers)):
        return -np.inf
        
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def objective(x):
    centers = x.reshape(N, 3)
    r = calculate_radius(centers)
    return -r

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_centers = None
    best_score = -np.inf
    
    # 4x4x4 grid
    ticks = np.linspace(-0.375, 0.375, 4)
    grid_coords = np.array(np.meshgrid(ticks, ticks, ticks)).T.reshape(-1, 3)
    
    num_restarts = 30
    
    for i in range(num_restarts):
        if time.time() - start_time > time_limit - 15:
            break
            
        np.random.seed(i + 123)
        idx = np.random.choice(64, 50, replace=False)
        x0 = grid_coords[idx].copy()
        
        jitter = 0.015 + 0.005 * (i % 4)
        x0 += np.random.normal(0, jitter, x0.shape)
        x0 = np.clip(x0, -0.499, 0.499)
        
        res = minimize(objective, x0, method='Powell', options={'maxiter': 1500, 'maxfev': 8000})
        
        if res.fun < best_score:
            best_score = res.fun
            best_centers = res.x.reshape(N, 3)

    if best_centers is None:
        # Fallback
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks_f = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks_f for y in ticks_f for x in ticks_f])
        best_centers = grid[:N]
        
    return best_centers
