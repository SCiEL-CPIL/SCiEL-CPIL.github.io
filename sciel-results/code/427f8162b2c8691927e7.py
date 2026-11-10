import numpy as np
import time
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 140.0 # Safety margin before 150s

def _score(centres):
    """Compute the radius for a given configuration."""
    # centres shape (N, 3)
    # Compute all pairwise distances
    # Using broadcasting for efficiency
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal to avoid zero distance issues in min
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # Cube is [-0.5, 0.5]
    # Distance to closest wall for point c is 0.5 - max(|cx|, |cy|, |cz|)
    abs_c = np.abs(centres)
    max_coord = np.max(abs_c, axis=1)
    wall_dists = 0.5 - max_coord
    
    min_wall_dist = np.min(wall_dists)
    
    radius = min(min_pair_dist / 2.0, min_wall_dist)
    return radius

def _objective(x, alpha=100.0):
    """
    Smooth objective function to minimize.
    We want to maximize the minimum distance.
    Minimizing sum of inverse powers of distances approximates this.
    """
    centres = x.reshape(N, 3)
    
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Clip distances to avoid division by zero
    min_dist_val = 1e-6
    safe_dists = np.maximum(dists, min_dist_val)
    
    # Wall distances
    abs_c = np.abs(centres)
    max_coord = np.max(abs_c, axis=1)
    wall_dists = 0.5 - max_coord
    safe_wall_dists = np.maximum(wall_dists, min_dist_val)
    
    # Objective: Sum of inverse powers
    # Higher power = sharper focus on the minimum
    p = 20
    obj = np.sum(safe_dists ** (-p)) + np.sum(safe_wall_dists ** (-p))
    
    return obj

def construct():
    start_time = time.time()
    best_centres = None
    best_score = -1.0
    
    # Initial guesses
    # 1. Random
    # 2. Perturbed Grid
    
    # Generate a base grid
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    base_grid = grid[:N]
    
    restarts = 0
    while time.time() - start_time < TIME_LIMIT:
        restarts += 1
        
        # Choose initialization
        if restarts % 2 == 1:
            # Perturbed grid
            noise = np.random.normal(0, 0.05, size=(N, 3))
            x0 = base_grid + noise
        else:
            # Random
            x0 = np.random.uniform(-0.45, 0.45, size=(N, 3))
        
        # Flatten for optimizer
        x_flat = x0.flatten()
        
        # Optimize
        try:
            res = minimize(_objective, x_flat, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
            if res.success:
                x_opt = res.x.reshape(N, 3)
                # Ensure points are strictly inside (clip if needed, though L-BFGS-B should keep them in bounds if we set bounds, but we didn't set bounds explicitly in minimize call, relying on objective repulsion. 
                # Wait, L-BFGS-B supports bounds. Let's use bounds to be safe and efficient.
                # Actually, the objective repels from walls strongly, so it should stay inside. 
                # But to be absolutely safe against numerical drift, we can clip.
                x_opt = np.clip(x_opt, -0.4999, 0.4999)
                
                score = _score(x_opt)
                
                if score > best_score:
                    best_score = score
                    best_centres = x_opt
        except Exception:
            pass
        
        # Check time
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
    if best_centres is None:
        # Fallback to grid
        return base_grid
        
    return best_centres
