import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _smooth_energy(params):
    """
    Calculate smooth energy to maximize minimum distances.
    """
    params = params.reshape(N, 2)
    # Pairwise distances
    diff = params[:, np.newaxis, :] - params[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, 1e-9)
    
    # Repulsive potential between circles
    pair_energy = np.sum(1.0 / (dists**2))
    
    # Boundary penalty
    radii = np.sqrt(np.sum(params**2, axis=1))
    dist_boundary = 1.0 - radii
    dist_boundary = np.maximum(dist_boundary, 1e-9)
    boundary_energy = np.sum(1.0 / (dist_boundary**2))
    
    return pair_energy + boundary_energy

def _max_radius(pos):
    """
    Calculate the maximum valid radius for a configuration.
    """
    pos = pos.reshape(N, 2)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    radii = np.sqrt(np.sum(pos**2, axis=1))
    min_boundary_dist = 1.0 - np.max(radii)
    
    return min(min_dist / 2.0, min_boundary_dist)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_pos = None
    best_r = 0.0
    
    # Multiple random restarts to escape local minima
    for i in range(100):
        if time.time() > deadline - 5:
            break
            
        np.random.seed(1000 + i)
        # Random initialization within a slightly smaller disk
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N)) * 0.8
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        init_pos = np.column_stack((x, y))
        
        def objective(x):
            return _smooth_energy(x)
            
        try:
            res = minimize(objective, init_pos, method='L-BFGS-B', 
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            opt_pos = res.x.reshape(N, 2)
            
            # Safety check: ensure points are strictly inside the unit circle
            max_r = np.max(np.sqrt(np.sum(opt_pos**2, axis=1)))
            if max_r >= 1.0:
                scale = 0.999 / max_r
                opt_pos = opt_pos * scale
            
            r = _max_radius(opt_pos)
            
            if r > best_r:
                best_r = r
                best_pos = opt_pos
        except Exception:
            continue
            
    if best_pos is None:
        # Fallback: return a simple grid if optimization fails
        # This is a very unlikely scenario
        return np.zeros((N, 2))
        
    return best_pos
