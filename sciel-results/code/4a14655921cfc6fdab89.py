"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Additive Guided Local Search penalty.
    
    This approach adds a constant penalty to the edges of the current local optimum tour.
    This forces the subsequent local search (2-opt) to find a different tour, as the current one
    is now artificially expensive. The penalty magnitude is scaled by the average edge length
    of the current tour to maintain scale invariance.
    """
    n = edge_distance.shape[0]
    
    # Create a copy of the distance matrix to avoid modifying the original
    penalized = edge_distance.copy()
    
    # Calculate the average edge length of the current local optimum tour
    # This provides a scale for the penalty
    tour_len = 0.0
    for i in range(n):
        j = local_opt_tour[i]
        k = local_opt_tour[(i + 1) % n]
        tour_len += edge_distance[i, j]
        
    avg_edge_len = tour_len / n if n > 0 else 1.0
    
    # Define the penalty amount
    # A factor of 0.5 means the penalty is half the average edge length.
    # This is a standard heuristic for GLS in TSP.
    penalty_factor = 0.5
    penalty = penalty_factor * avg_edge_len
    
    # Add the penalty to all edges in the current local optimum tour
    for i in range(n):
        j = local_opt_tour[i]
        k = local_opt_tour[(i + 1) % n]
        penalized[i, j] += penalty
        penalized[j, i] += penalty
        penalized[i, k] += penalty
        penalized[k, i] += penalty
        
    # Ensure symmetry (should be preserved, but good practice)
    penalized = (penalized + penalized.T) / 2.0
    
    # Ensure diagonal is zero
    np.fill_diagonal(penalized, 0.0)
    
    # Ensure all values are finite
    penalized = np.where(np.isfinite(penalized), penalized, edge_distance)
    
    return penalized
# EVOLVE-BLOCK-END
