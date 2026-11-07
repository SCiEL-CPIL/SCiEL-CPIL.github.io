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
    
    # Standard Guided Local Search (GLS) approach:
    # Add a penalty to edges based on their frequency of occurrence in
    # previously found local optima. This modifies the landscape to
    # discourage the search from returning to visited basins.
    
    # Tune alpha to balance the penalty against the typical edge lengths.
    # Edge lengths in unit square are typically between 0 and ~1.4.
    # edge_n_used max is 20.
    # If alpha=0.1, max penalty is 2.0, which is strong.
    # If alpha=0.05, max penalty is 1.0.
    # Let's use 0.1 for a strong push.
    alpha = 0.1
    
    # Calculate the penalty matrix
    # We apply the penalty to all edges, weighted by their usage count.
    # This is more effective than only penalizing the current tour edges
    # because it prevents the 2-opt from oscillating between nearby
    # local optima.
    penalty = alpha * edge_n_used
    
    # Add penalty to the true distances
    penalized = edge_distance + penalty
    
    # Ensure the result is symmetric (edge_n_used is symmetric, edge_distance is symmetric)
    # Just in case of numerical drift, we can symmetrize, but it's likely already symmetric.
    # penalized = (penalized + penalized.T) / 2.0
    
    # Ensure finiteness and positivity
    penalized = np.where(np.isfinite(penalized), penalized, edge_distance)
    penalized = np.maximum(penalized, 1e-10)
    
    return penalized
# EVOLVE-BLOCK-END
