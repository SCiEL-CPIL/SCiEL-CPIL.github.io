"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial.distance import pdist, squareform

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time

def compute_score(centers):
    """Compute the radius for a given set of centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds
    min_coord = np.min(centers, axis=0)
    max_coord = np.max(centers, axis=0)
    if np.any(min_coord < -0.5) or np.any(max_coord > 0.5):
        return 0.0
    
    # Compute pairwise distances
    dists = pdist(centers)
    min_dist = np.min(dists)
    
    # Compute distance to walls
    wall_dists = np.min(np.stack([
        centers + 0.5,
        0.5 - centers
    ], axis=1), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    r = 0.5 * min(min_dist, min_wall_dist)
    return r

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_centers = None
    best_score = 0.0
    
    # Initial grid solution
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial_centers = grid[:N]
    best_centers = initial_centers.copy()
    best_score = compute_score(best_centers)
    
    # Multi-start local optimization using L-BFGS-B
    np.random.seed(42)
    
    def objective_and_grad(centers_flat):
        centers = centers_flat.reshape(N, 3)
        # Compute pairwise distances
        dists = pdist(centers)
        min_dist = np.min(dists)
        
        # Compute distance to walls
        wall_dists = np.min(np.stack([
            centers + 0.5,
            0.5 - centers
        ], axis=1), axis=1)
        min_wall_dist = np.min(wall_dists)
        
        r = 0.5 * min(min_dist, min_wall_dist)
        
        # We want to maximize r, so minimize -r
        # Gradient is tricky with min, use subgradient approximation
        # Use a smooth approximation: soft-min
        eps = 1e-6
        all_dists = np.concatenate([dists, wall_dists])
        soft_min = -np.logaddexp.reduce(np.log(eps) + np.log(-all_dists + 2*eps)) / 1.0
        # Actually, let's just use the raw value and let L-BFGS handle it
        return -r, None
    
    # Use a different approach: maximize the minimum distance directly
    # We'll use a penalty-based approach with L-BFGS-B
    
    def objective_penalty(centers_flat, penalty=100.0):
        centers = centers_flat.reshape(N, 3)
        
        # Pairwise distances
        dists = pdist(centers)
        min_dist = np.min(dists)
        
        # Wall distances
        wall_dists = np.min(np.stack([
            centers + 0.5,
            0.5 - centers
        ], axis=1), axis=1)
        min_wall_dist = np.min(wall_dists)
        
        r = 0.5 * min(min_dist, min_wall_dist)
        
        # Penalty for being outside bounds
        penalty_val = 0.0
        for i in range(N):
            for j in range(3):
                if centers[i, j] < -0.5:
                    penalty_val += penalty * (-0.5 - centers[i, j]) ** 2
                elif centers[i, j] > 0.5:
                    penalty_val += penalty * (centers[i, j] - 0.5) ** 2
        
        return -r + penalty_val
    
    # Try multiple random starts
    n_starts = 8
    for s in range(n_starts):
        if time.time() > deadline:
            break
        
        if s == 0:
            centers = initial_centers.copy()
        else:
            # Random perturbation
            centers = initial_centers.copy()
            noise_scale = 0.05 / (s + 1)
            centers += np.random.randn(N, 3) * noise_scale
            # Clip to valid range
            centers = np.clip(centers, -0.49, 0.49)
        
        # Optimize
        try:
            result = minimize(
                objective_penalty,
                centers.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            opt_centers = result.x.reshape(N, 3)
            # Ensure centers are within bounds
            opt_centers = np.clip(opt_centers, -0.5, 0.5)
            
            score = compute_score(opt_centers)
            if score > best_score:
                best_score = score
                best_centers = opt_centers.copy()
                
        except Exception as e:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 20:
        try:
            result = minimize(
                objective_penalty,
                best_centers.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            opt_centers = np.clip(result.x.reshape(N, 3), -0.5, 0.5)
            score = compute_score(opt_centers)
            if score > best_score:
                best_score = score
                best_centers = opt_centers.copy()
        except Exception as e:
            pass
    
    return best_centers
# EVOLVE-BLOCK-END
