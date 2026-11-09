import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 145.0 # Leave 5s buffer
    
    def objective(params):
        points = params.reshape(8, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        iu = np.triu_indices(8, k=1)
        d_pairs = dists[iu]
        
        if np.any(d_pairs < 1e-12):
            return 1e10
        
        D = np.max(d_pairs)
        
        C = 28
        val = np.sum(np.log(d_pairs)) - C * np.log(D)
        return -val
    
    # Initial guess: Regular octagon
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    x_init = np.cos(angles)
    y_init = np.sin(angles)
    init_params = np.column_stack((x_init, y_init)).flatten()
    
    best_val = objective(init_params)
    best_params = init_params.copy()
    
    # Try different perturbations
    np.random.seed(123)
    
    # Strategy 1: Small noise
    for i in range(20):
        if time.time() > deadline: break
        noise = np.random.normal(0, 0.05, 16)
        params = init_params + noise
        cx = np.mean(params[0::2])
        cy = np.mean(params[1::2])
        params[0::2] -= cx
        params[1::2] -= cy
        
        res = minimize(objective, params, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x

    # Strategy 2: Symmetry breaking - move opposite points
    # e.g. points 0 and 4
    for scale in [0.1, 0.2, 0.3, 0.5]:
        if time.time() > deadline: break
        params = init_params.copy()
        # Move point 0 (angle 0) and point 4 (angle pi)
        # Point 0 is at (1,0), Point 4 is at (-1,0)
        # Move them along x axis
        params[0] += scale  # x0
        params[1] -= scale  # x4 (index 1 is y0? No, flatten is x0,y0,x1,y1...)
        # Indices: 0:x0, 1:y0, 2:x1, 3:y1, ..., 8:x4, 9:y4
        params[0] = 1.0 + scale
        params[8] = -1.0 - scale
        
        cx = np.mean(params[0::2])
        cy = np.mean(params[1::2])
        params[0::2] -= cx
        params[1::2] -= cy
        
        res = minimize(objective, params, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x

    points = best_params.reshape(8, 2)
    
    # Normalize to diameter 2
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    D = np.max(dists)
    if D > 0:
        points = points * (2.0 / D)
        
    return points.tolist()
