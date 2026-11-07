"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation using multiplicative penalties.
    
    Instead of adding a penalty, this approach multiplies the cost of edges
    in the current tour by a factor that increases with their usage count.
    This changes the relative weights of edges, potentially allowing the
    local search to escape local optima by making frequently used edges
    significantly more expensive relative to others.
    """
    n = edge_distance.shape[0]
    perturbed = edge_distance.copy()
    
    # Get the edges in the current local optimal tour
    tour = local_opt_tour
    tour_edges = []
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        if u < v:
            tour_edges.append((u, v))
        else:
            tour_edges.append((v, u))
            
    # Apply multiplicative penalty to each edge in the tour
    # The factor grows linearly with the usage count.
    # alpha = 0.05 means an edge used 20 times will have its cost doubled.
    alpha = 0.05
    for (u, v) in tour_edges:
        count = edge_n_used[u, v]
        if count > 0:
            factor = 1.0 + alpha * count
            perturbed[u, v] *= factor
            perturbed[v, u] *= factor
    
    # Ensure all values are finite and non-negative
    # Multiplication of positive distances by positive factors should keep them positive and finite
    # unless input was invalid, but we handle it just in case.
    perturbed = np.maximum(perturbed, 0.0)
    perturbed = np.where(np.isfinite(perturbed), perturbed, 1.0)
    
    return perturbed
# EVOLVE-BLOCK-END
