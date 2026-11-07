"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation: apply context-aware multiplicative penalties
    to edges in the current local optimal tour, scaled by their usage count.
    
    Only tour edges are penalized, preserving global geometry for 2-opt moves
    while specifically breaking frequently-used 'stuck' edges.
    """
    n = edge_distance.shape[0]
    perturbed = edge_distance.copy()
    
    # Extract tour edges (each edge appears twice in the tour list)
    tour_edges = set()
    for i in range(len(local_opt_tour)):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % len(local_opt_tour)]
        if u < v:
            tour_edges.add((u, v))
        else:
            tour_edges.add((v, u))
    
    # Compute penalty for each tour edge based on usage count
    # Use a smooth saturating function: penalty grows with usage but with diminishing returns
    # beta controls max penalty magnitude, gamma controls how quickly it saturates
    beta = 0.4
    gamma = 0.3
    
    max_usage = edge_n_used.max() if edge_n_used.size > 0 else 0
    if max_usage <= 0:
        return perturbed
    
    # For each tour edge, apply multiplicative penalty
    for (u, v) in tour_edges:
        count = edge_n_used[u, v]
        if count > 0:
            # Smooth penalty: 1 + beta * (1 - exp(-gamma * count))
            # This gives ~0.12 penalty after 1 use, ~0.30 after 5 uses, ~0.37 after 10 uses
            penalty_factor = 1.0 + beta * (1.0 - np.exp(-gamma * count))
            perturbed[u, v] *= penalty_factor
            perturbed[v, u] *= penalty_factor
    
    # Add a very small global exploration penalty to encourage diversity
    # This is tiny to not distort the geometry significantly
    global_penalty = 0.01 * (edge_n_used / (1.0 + max_usage))
    perturbed *= (1.0 + global_penalty)
    
    # Ensure symmetry and finiteness
    perturbed = (perturbed + perturbed.T) / 2.0
    perturbed = np.nan_to_num(perturbed, nan=0.0, posinf=1e6, neginf=0.0)
    
    return perturbed
# EVOLVE-BLOCK-END
