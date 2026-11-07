"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided Local Search penalty for TSP.
    
    Returns a modified distance matrix where edges in the current local optimum
    tour receive a penalty proportional to their accumulated usage count.
    
    The penalty is scaled relative to the median edge distance to be meaningful
    across instances. We use an adaptive penalty that grows with usage but
    saturates to avoid over-penalizing.
    """
    n = edge_distance.shape[0]
    
    # Compute scale factor: median of non-zero off-diagonal distances
    mask = ~np.eye(n, dtype=bool)
    nonzero_mask = (edge_distance > 1e-10) & mask
    if np.any(nonzero_mask):
        dist_scale = np.median(edge_distance[nonzero_mask])
    else:
        dist_scale = 1.0
    
    # Identify edges in the current local optimum tour
    tour_edges = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = local_opt_tour[i]
        k = local_opt_tour[(i + 1) % n]
        # The tour is a sequence of nodes; edges are between consecutive nodes
        # Actually, local_opt_tour is a permutation, so edges are (tour[i], tour[i+1])
        pass
    
    # Re-interpret: local_opt_tour is a list of node indices in tour order
    # Edges are (local_opt_tour[i], local_opt_tour[i+1]) for i in range(n-1)
    # and (local_opt_tour[n-1], local_opt_tour[0])
    for i in range(n - 1):
        u = local_opt_tour[i]
        v = local_opt_tour[i + 1]
        tour_edges[u, v] = 1.0
        tour_edges[v, u] = 1.0
    # Last edge
    u = local_opt_tour[n - 1]
    v = local_opt_tour[0]
    tour_edges[u, v] = 1.0
    tour_edges[v, u] = 1.0
    
    # Compute penalty for each edge in the tour based on its usage count
    # Use a smooth penalty: penalty = alpha * dist_scale * (usage / (1 + usage))
    # This saturates at alpha * dist_scale for heavily used edges
    alpha = 0.5  # Base penalty strength
    
    # Get usage counts for tour edges
    usage_in_tour = edge_n_used * tour_edges
    
    # Compute penalty matrix (only non-zero for tour edges)
    # penalty_ij = alpha * dist_scale * (n_used_ij / (1 + n_used_ij)) for tour edges
    penalty = alpha * dist_scale * (usage_in_tour / (1.0 + usage_in_tour))
    
    # Return modified distance matrix
    result = edge_distance + penalty
    
    # Ensure symmetry and non-negativity
    result = (result + result.T) / 2.0
    result = np.maximum(result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
