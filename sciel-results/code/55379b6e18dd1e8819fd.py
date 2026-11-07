"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_GLOBAL_START_TIME = time.time()
_WALL_CLOCK_LIMIT = 320.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Check time limit - return original distances if we're running out of time
    if time.time() - _GLOBAL_START_TIME > _WALL_CLOCK_LIMIT:
        return edge_distance.copy()
    
    n = edge_distance.shape[0]
    
    # Use additive penalty (standard GLS approach)
    # The penalty should be proportional to edge usage count
    # Tune alpha to balance exploration vs exploitation
    alpha = 0.5
    
    # Apply penalty only to edges that are in the current tour
    # This focuses the perturbation on breaking the current solution
    tour_edges = np.zeros_like(edge_distance)
    for i in range(n):
        j = local_opt_tour[i]
        k = local_opt_tour[(i + 1) % n]
        tour_edges[i, j] = 1.0
        tour_edges[j, i] = 1.0
    
    # Additive penalty: penalize used edges based on their count
    # This encourages the search to avoid recently used edges
    penalized = edge_distance + alpha * edge_n_used * tour_edges
    
    # Ensure the result is finite and positive
    penalized = np.where(np.isfinite(penalized), penalized, edge_distance)
    penalized = np.maximum(penalized, 1e-10)
    
    return penalized
# EVOLVE-BLOCK-END
