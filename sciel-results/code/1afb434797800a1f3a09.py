"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

_start_time = time.time()

def _time_left():
    return DEADLINE - (time.time() - _start_time)

def _score(centers):
    """Compute the maximum common radius for given centers in [-0.5, 0.5]^3."""
    if centers.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(centers)):
        return -1.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return -1.0
    
    # Wall distance
    wall_dist = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_dist)
    
    # Pairwise distances
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    
    return min(min_wall, min_pair)

def _neg_score(centers):
    """Negative score for minimization."""
    return -_score(centers)

def _make_grid(m, n_use):
    """Create a grid of m^3 points in [-0.5, 0.5]^3 and return first n_use."""
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n_use]

def construct():
    global _start_time
    _start_time = time.time()
    
    # Initial grid-based solution
    m = int(np.ceil(N ** (1.0/3.0) - 1e-9))
    best_centers = _make_grid(m, N)
    best_score = _score(best_centers)
    
    # Try different initial configurations and local optimization
    configs = []
    
    # 1. Simple cubic grid
    configs.append(best_centers.copy())
    
    # 2. Slightly perturbed grid
    np.random.seed(42)
    perturbed = best_centers + np.random.uniform(-0.01, 0.01, (N, 3))
    # Clamp to cube
    perturbed = np.clip(perturbed, -0.49, 0.49)
    configs.append(perturbed)
    
    # 3. Random initialization
    for seed in [123, 456, 789, 101, 202, 303]:
        np.random.seed(seed)
        rand_centers = np.random.uniform(-0.4, 0.4, (N, 3))
        configs.append(rand_centers)
    
    # 4. Face-centered arrangement attempt
    # Place points in layers
    layer_configs = []
    # Try 5 layers of 10 points each (roughly)
    for layer_seed in [11, 22, 33]:
        np.random.seed(layer_seed)
        centers = np.zeros((N, 3))
        points_per_layer = [10, 10, 10, 10, 10]
        idx = 0
        for layer_i, n_pts in enumerate(points_per_layer):
            z = -0.4 + 0.8 * layer_i / 4.0
            for j in range(n_pts):
                # Distribute in a 5x2 grid within the layer
                x = -0.4 + 0.8 * (j % 5) / 4.0
                y = -0.4 + 0.8 * (j // 5) / 1.0
                centers[idx] = [x, y, z]
                idx += 1
        centers += np.random.uniform(-0.02, 0.02, (N, 3))
        centers = np.clip(centers, -0.49, 0.49)
        layer_configs.append(centers)
    
    configs.extend(layer_configs)
    
    # Local optimization using Nelder-Mead or Powell
    def _local_optimize(centers, method='Nelder-Mead', maxiter=5000):
        """Perform local optimization on centers."""
        if _time_left() < 5.0:
            return centers, _score(centers)
        
        # Use Powell method which doesn't require gradients
        try:
            result = minimize(
                _neg_score, 
                centers, 
                method=method,
                options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            new_centers = result.x
            new_score = _score(new_centers)
            if new_score > 0:
                return new_centers, new_score
            return centers, _score(centers)
        except:
            return centers, _score(centers)
    
    # Try local optimization on each config
    for i, config in enumerate(configs):
        if _time_left() < 10.0:
            break
        
        centers, score = _local_optimize(config, 'Powell', maxiter=3000)
        if score > best_score:
            best_score = score
            best_centers = centers
            print(f"Config {i}: improved to {score:.10f}")
    
    # If time remains, do more refined optimization
    if _time_left() > 20.0:
        # Multiple rounds of Powell with tighter tolerances
        for round_num in range(3):
            if _time_left() < 10.0:
                break
            centers, score = _local_optimize(best_centers, 'Powell', maxiter=5000)
            if score > best_score:
                best_score = score
                best_centers = centers
                print(f"Round {round_num}: improved to {score:.10f}")
    
    # Final check
    final_score = _score(best_centers)
    if final_score > best_score:
        best_score = final_score
    
    return best_centers
# EVOLVE-BLOCK-END
