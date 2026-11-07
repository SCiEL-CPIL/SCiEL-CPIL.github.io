"""Guided local search penalty rule for the TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation: add targeted penalties to edges in the
    current local optimal tour, scaled by a smooth function of their usage count.
    
    Design:
    - Only penalize edges that are in the current tour (targeted, not global)
    - Penalty magnitude grows with usage count using a power-law function
    - Scale penalty relative to the mean edge distance for robustness
    - Return a finite matrix of the same shape
    """
    n = edge_distance.shape[0]
    result = edge_distance.copy()
    
    # Extract edges in the current tour
    tour_edges = []
    for i in range(n):
        j = local_opt_tour[i]
        tour_edges.append((i, j))
    
    # Compute mean edge distance for scaling
    mean_dist = np.mean(edge_distance)
    
    # For each edge in the tour, add a penalty proportional to its usage count
    # Using a power-law scaling to emphasize persistent edges
    for i, j in tour_edges:
        count = edge_n_used[i, j]
        if count > 0:
            # Power-law penalty: grows faster than linear for heavily used edges
            penalty = 0.5 * mean_dist * (count ** 1.5) / (1.0 + count)
            result[i, j] += penalty
            result[j, i] += penalty
    
    # Ensure all values are finite
    result = np.where(np.isfinite(result), result, edge_distance)
    
    return result
# EVOLVE-BLOCK-END
