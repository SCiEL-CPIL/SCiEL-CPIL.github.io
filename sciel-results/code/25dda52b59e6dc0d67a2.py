import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_radius(centers):
    dists = np.min(np.abs(centers), axis=1)
    wall_dist = 0.5 - dists
    r_wall = np.min(wall_dist)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_c = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists_c, np.inf)
    r_circle = np.min(dists_c) / 2.0
    
    return min(r_wall, r_circle)

def neg_score(centers):
    c = centers.reshape(N, 2)
    if np.any(np.abs(c) > 0.5):
        return 1000.0
    r = compute_radius(c)
    return -r

def construct():
    start_time = time.time()
    deadline = 110.0
    
    best_radius = -np.inf
    best_centers = None
    
    grid_coords = np.linspace(-0.5, 0.5, 7)
    gx, gy = np.meshgrid(grid_coords, grid_coords)
    grid_points = np.column_stack((gx.ravel(), gy.ravel()))
    
    positions_50 = [
        np.array([1/14.0, 1/14.0]),
        np.array([-1/14.0, 1/14.0]),
        np.array([0.0, 0.0]),
        np.random.uniform(-0.5, 0.5, 2)
    ]
    
    initial_configs = []
    for p50 in positions_50:
        init = np.vstack([grid_points, p50])
        initial_configs.append(init)
    initial_configs.append(np.random.uniform(-0.5, 0.5, (N, 2)))
    
    for i, init in enumerate(initial_configs):
        if time.time() > deadline:
            break
            
        time_limit = 20.0
        stop_flag = [False]
        def callback(xk):
            if time.time() > start_time + time_limit:
                stop_flag[0] = True
            return stop_flag[0]
            
        x0 = init.flatten()
        
        try:
            res = minimize(neg_score, x0, method='Powell', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10},
                           callback=callback)
            
            current_centers = res.x.reshape(N, 2)
            current_radius = compute_radius(current_centers)
            
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = current_centers.copy()
                
        except Exception:
            pass
            
    # Polish phase
    if best_centers is not None and time.time() < deadline - 10:
        time_limit = deadline - time.time() - 5
        stop_flag = [False]
        def callback_polish(xk):
            if time.time() > start_time + time_limit:
                stop_flag[0] = True
            return stop_flag[0]
            
        try:
            res = minimize(neg_score, best_centers.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14},
                           callback=callback_polish)
            current_centers = res.x.reshape(N, 2)
            current_radius = compute_radius(current_centers)
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = current_centers.copy()
        except Exception:
            pass

    if best_centers is None:
        best_centers = initial_configs[0]
        
    return best_centers
