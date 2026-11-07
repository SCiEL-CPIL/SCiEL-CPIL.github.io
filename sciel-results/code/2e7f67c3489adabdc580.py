"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided Local Search penalty for TSP.
    
    Applies a smooth, saturating penalty to the edges present in the current
    local optimal tour. The magnitude of the penalty depends on the historical
    usage count of that edge, discouraging the reuse of edges that have been
    part of previous local optima.
    """
    # Start with a copy of the true distances
    new_dist = edge_distance.copy()
    
    n = len(local_opt_tour)
    if n < 3:
        return new_dist

    # Penalty parameters
    # alpha: Max penalty magnitude (saturation level)
    # beta: Rate of saturation
    alpha = 0.3
    beta = 0.5
    
    # Iterate over edges in the current local optimal tour
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        
        # Get the number of times this edge has been used in previous local optima
        count = edge_n_used[u, v]
        
        if count > 0:
            # Calculate saturating penalty
            # As count increases, penalty approaches alpha
            penalty = alpha * (1.0 - np.exp(-beta * count))
            
            # Apply penalty to the distance matrix (symmetric)
            new_dist[u, v] += penalty
            new_dist[v, u] += penalty

    return new_dist
# EVOLVE-BLOCK-END
