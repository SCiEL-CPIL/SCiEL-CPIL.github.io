"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START

def _score(centers):
    """Compute the packing radius for given centers in [-0.5, 0.5]^2."""
    # Wall constraints
    min_x = 0.5 - centers[:, 0].max()
    min_y = 0.5 - centers[:, 1].max()
    max_x = centers[:, 0].min() + 0.5
    max_y = centers[:, 1].min() + 0.5
    wall_r = min(min_x, min_y, max_x, max_y)
    
    if wall_r <= 0:
        return 0.0
    
    # Pairwise distances
    # Compute all pairwise distances efficiently
    n = centers.shape[0]
    # Use broadcasting for pairwise distances (memory intensive for large N, but N=50 is fine)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    
    return min(wall_r, min_dist / 2.0)


def _negative_score(centers_flat):
    """Objective function to minimize (negative of score)."""
    centers = centers_flat.reshape(-1, 2)
    return -_score(centers)


def _initial_grid():
    """Create an initial grid layout."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _random_init():
    """Random initialization within the square."""
    return np.random.uniform(-0.45, 0.45, size=(N, 2))


def _optimize_from(start, max_iter=5000, tol=1e-12):
    """Run optimization from a starting point."""
    start_flat = start.flatten()
    
    # Use L-BFGS-B for efficiency
    result = minimize(
        _negative_score,
        start_flat,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10}
    )
    return result.x.reshape(-1, 2)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Start with a grid
    best_centers = _initial_grid()
    best_score = _score(best_centers)
    
    # Try multiple random restarts
    rng = np.random.default_rng(42)
    n_restarts = 0
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30.0:  # Need at least 30s for one more optimization
            break
        
        # Random initialization
        init = _random_init()
        n_restarts += 1
        
        # Optimize
        try:
            optimized = _optimize_from(init, max_iter=3000, tol=1e-12)
            score = _score(optimized)
            
            if score > best_score:
                best_score = score
                best_centers = optimized
        except Exception:
            pass
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            break
    
    # Final polish with higher precision if time allows
    elapsed = time.time() - start_time
    if elapsed < deadline - 2.0:
        try:
            polished = _optimize_from(best_centers, max_iter=10000, tol=1e-14)
            score = _score(polished)
            if score > best_score:
                best_score = score
                best_centers = polished
        except Exception:
            pass
    
    # Ensure all centers are strictly inside the square
    # Clip to be safe
    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    
    return best_centers

# EVOLVE-BLOCK-END
