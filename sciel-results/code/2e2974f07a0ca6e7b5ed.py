"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search using an additive penalty.
    
    Args:
        edge_distance: (N, N) array of true Euclidean distances.
        local_opt_tour: (N,) array of node indices forming the current tour.
        edge_n_used: (N, N) array of symmetric counts of edge usage.
        
    Returns:
        (N, N) array of perturbed distances.
    """
    n = edge_distance.shape[0]
    if n <= 1:
        return edge_distance.copy()

    # Calculate scale based on the median of non-zero distances in the matrix.
    # This provides a robust measure of typical edge length.
    lower_tri = np.tril_indices(n, k=-1)
    distances = edge_distance[lower_tri]
    non_zero_dist = distances[distances > 0]
    
    if len(non_zero_dist) > 0:
        scale = np.median(non_zero_dist)
    else:
        scale = 1.0

    # Additive penalty coefficient.
    # We add alpha * count to the distance.
    # If an edge is used k times, its cost increases by k * alpha.
    # Choosing alpha as a fraction of the median distance ensures the penalty
    # is significant relative to the problem scale.
    alpha = 0.5 * scale

    # Apply the additive penalty.
    perturbed_distance = edge_distance + alpha * edge_n_used
    
    # Ensure no negative or infinite values
    perturbed_distance = np.nan_to_num(perturbed_distance, nan=0.0, posinf=1e6, neginf=0.0)
    
    return perturbed_distance
# EVOLVE-BLOCK-END
