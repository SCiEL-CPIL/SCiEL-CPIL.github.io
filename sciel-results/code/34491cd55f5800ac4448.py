<<<<<<< SEARCH
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Simple cubic grid in the side-1 cube centred at the origin: valid but improvable.
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]
# EVOLVE-BLOCK-END
=======
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(centers):
    """Compute the largest common radius for given centers."""
    # Check bounds: all centers must be within [-0.5, 0.5] in each dimension
    if not np.all(np.isfinite(centers)):
        return -1.0
    if centers.shape != (N, 3):
        return -1.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return -1.0
    
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Minimum distance between any pair (off-diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Minimum distance from any center to any wall
    min_wall_dist = np.min(0.5 - np.abs(centers))
    
    # The radius is half of the minimum of these two
    return min(min_pair_dist, min_wall_dist) / 2.0


def _init_grid():
    """Initialize with a simple cubic grid."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def construct():
    # Start with the simple cubic grid
    best_centers = _init_grid()
    best_score = _score(best_centers)
    
    # Deadline guard: 150s wall-clock limit, use 145s for safety
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Use a local optimization approach: iterative relaxation
    # For each sphere, try to move it to increase the minimum distance
    # This is a simplified version of what more sophisticated packers do
    
    centers = best_centers.copy()
    n_iter = 0
    max_iter = 500000
    
    while time.time() < deadline and n_iter < max_iter:
        # Pick a random sphere
        i = np.random.randint(N)
        
        # Compute current minimum distance constraints for sphere i
        # Distance to other spheres
        diffs = centers[i] - centers
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        np.fill_diagonal(dists, np.inf)
        
        # Distance to walls
        wall_dists = np.array([
            0.5 - centers[i][0],
            0.5 + centers[i][0],
            0.5 - centers[i][1],
            0.5 + centers[i][1],
            0.5 - centers[i][2],
            0.5 + centers[i][2]
        ])
        
        current_min = min(np.min(dists), np.min(wall_dists))
        
        # Try small random moves
        improvement = False
        for attempt in range(20):
            # Random small displacement
            delta = np.random.normal(0, 0.01, 3)
            new_center = centers[i] + delta
            
            # Check if still within bounds
            if np.any(np.abs(new_center) > 0.5):
                continue
            
            # Tentatively move
            old_center = centers[i].copy()
            centers[i] = new_center
            
            # Compute new minimum distance for this sphere
            diffs = centers[i] - centers
            dists = np.sqrt(np.sum(diffs * diffs, axis=1))
            np.fill_diagonal(dists, np.inf)
            
            wall_dists = np.array([
                0.5 - centers[i][0],
                0.5 + centers[i][0],
                0.5 - centers[i][1],
                0.5 + centers[i][1],
                0.5 - centers[i][2],
                0.5 + centers[i][2]
            ])
            
            new_min = min(np.min(dists), np.min(wall_dists))
            
            # Compute overall score
            # We need to check if this move improves the global minimum
            # For efficiency, we can just check if new_min > current_min
            if new_min > current_min:
                improvement = True
                break
            else:
                # Revert
                centers[i] = old_center
        
        if not improvement:
            # Try a larger random move
            delta = np.random.normal(0, 0.05, 3)
            new_center = centers[i] + delta
            if np.all(np.abs(new_center) <= 0.5):
                old_center = centers[i].copy()
                centers[i] = new_center
                # Don't revert; accept if it doesn't make things too bad
                # This helps escape local minima
        
        n_iter += 1
        
        # Periodically check the score
        if n_iter % 1000 == 0:
            current_score = _score(centers)
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
    
    # Final score check
    final_score = _score(centers)
    if final_score > best_score:
        best_centers = centers.copy()
    
    return best_centers
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
