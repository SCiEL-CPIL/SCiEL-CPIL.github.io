"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Args:
        edge_distance: true Euclidean distance matrix (n x n)
        local_opt_tour: current tour as a sequence of node indices
        edge_n_used: symmetric count matrix of how often each edge has been used
    
    Returns:
        Perturbed distance matrix with penalties applied to frequently used edges.
    """
    n = edge_distance.shape[0]
    
    # Normalize edge usage counts
    max_used = edge_n_used.max()
    if max_used > 0:
        normalized_usage = edge_n_used / max_used
    else:
        normalized_usage = np.zeros_like(edge_n_used)
    
    # Identify edges in the current local optimum tour
    tour_edges = np.zeros((n, n), dtype=bool)
    for i in range(len(local_opt_tour)):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % len(local_opt_tour)]
        tour_edges[u, v] = True
        tour_edges[v, u] = True
    
    # Create penalty: stronger for edges in current tour AND frequently used
    # Use a combination: base penalty scaled by usage, boosted for current tour edges
    base_penalty = 0.15 * normalized_usage
    tour_boost = 0.10 * tour_edges
    
    # Combined penalty multiplier
    penalty_multiplier = 1.0 + base_penalty + tour_boost
    
    # Apply penalty only to off-diagonal entries
    perturbed = edge_distance * penalty_multiplier
    
    # Ensure diagonal stays zero
    np.fill_diagonal(perturbed, 0.0)
    
    # Ensure finite values
    perturbed = np.where(np.isfinite(perturbed), perturbed, edge_distance)
    
    return perturbed
# EVOLVE-BLOCK-END
