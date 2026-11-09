"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search using a multiplicative penalty.
    
    Args:
        edge_distance: (N, N) array of true Euclidean distances.
        local_opt_tour: (N,) array of node indices forming the current tour.
        edge_n_used: (N, N) array of symmetric counts of edge usage.
        
    Returns:
        (N, N) array of perturbed distances.
    """
    # Calculate a robust scale factor for the distances.
    # Using the median of the lower triangle (excluding zeros) is robust to outliers.
    n = edge_distance.shape[0]
    if n > 1:
        # Get all non-zero distances
        lower_tri = np.tril_indices(n, k=-1)
        distances = edge_distance[lower_tri]
        # Filter out any zeros just in case, though Euclidean dist should be > 0 for distinct points
        non_zero_dist = distances[distances > 0]
        if len(non_zero_dist) > 0:
            scale = np.median(non_zero_dist)
        else:
            scale = 1.0
    else:
        scale = 1.0

    # Multiplicative penalty coefficient.
    # We want to increase the cost of used edges.
    # If an edge is used k times, its cost is multiplied by (1 + alpha * k).
    # For k=5, we want a significant increase, say 50% to 100%.
    # alpha = 0.2 means k=5 -> 1 + 1.0 = 2.0 (100% increase).
    # alpha = 0.1 means k=5 -> 1 + 0.5 = 1.5 (50% increase).
    # Let's use 0.15 for a balance.
    alpha = 0.15

    # Apply the multiplicative penalty.
    # perturbed_distance = edge_distance * (1 + alpha * edge_n_used)
    # This ensures that shorter edges remain relatively cheaper than longer ones,
    # unlike additive penalties which can invert the order if the penalty is large.
    perturbed_distance = edge_distance * (1.0 + alpha * edge_n_used)
    
    # Ensure no negative or infinite values (though mathematically it shouldn't happen)
    perturbed_distance = np.nan_to_num(perturbed_distance, nan=0.0, posinf=1e6, neginf=0.0)
    
    return perturbed_distance
# EVOLVE-BLOCK-END
