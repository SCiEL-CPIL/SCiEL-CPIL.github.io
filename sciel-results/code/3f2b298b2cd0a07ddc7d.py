import numpy as np
import time

_GLOBAL_START_TIME = time.time()
_WALL_CLOCK_LIMIT = 320.0

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    if time.time() - _GLOBAL_START_TIME > _WALL_CLOCK_LIMIT:
        return edge_distance.copy()
    
    n = edge_distance.shape[0]
    
    # Calculate average edge length for scaling
    # Use the current tour's length as a proxy for scale, or global mean
    # Global mean is more stable
    # Mask out diagonal
    mask = ~np.eye(n, dtype=bool)
    mean_dist = np.mean(edge_distance[mask]) if n > 1 else 1.0
    
    # If no edges have been used yet, return original
    if edge_n_used.max() == 0:
        return edge_distance.copy()
    
    # Structural Change: Use un-normalized usage counts with a tuned additive penalty.
    # This allows the penalty magnitude to grow with the number of iterations,
    # increasing pressure to escape local optima in later rounds.
    
    # Coefficient alpha
    # If max count is ~20, and we want max penalty to be ~0.5 * mean_dist
    # alpha * 20 = 0.5 * mean_dist  => alpha = 0.025 * mean_dist
    alpha = 0.025 * mean_dist
    
    # Calculate penalty
    penalty = alpha * edge_n_used
    
    # Add to distance matrix
    penalized = edge_distance + penalty
    
    # Ensure finiteness and symmetry
    penalized = np.where(np.isfinite(penalized), penalized, edge_distance)
    penalized = (penalized + penalized.T) / 2.0
    
    return penalized
