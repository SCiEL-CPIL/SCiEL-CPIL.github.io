import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    n = edge_distance.shape[0]
    # Copy the matrix to avoid side effects
    perturbed = edge_distance.copy()
    
    # Calculate the mean edge length of the current tour
    # local_opt_tour is a list of indices [v0, v1, ..., v_{n-1}]
    # Edges are (v0, v1), (v1, v2), ..., (v_{n-1}, v0)
    
    # Get the edges of the current tour
    tour_edges = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges.append((u, v))
        
    # Calculate mean edge length in the current tour
    if len(tour_edges) > 0:
        tour_lengths = [edge_distance[u, v] for u, v in tour_edges]
        mean_edge_len = np.mean(tour_lengths)
    else:
        mean_edge_len = 1.0 # Fallback, shouldn't happen for n>=2
        
    # Penalty parameter
    # We want the penalty to be significant but not dominant.
    # A common choice is a small fraction of the mean edge length.
    # Let's try 0.1 * mean_edge_len per usage count.
    penalty_scale = 0.1 * mean_edge_len
    
    # Apply penalty to edges in the current tour
    for u, v in tour_edges:
        count = edge_n_used[u, v]
        # If count is 0, no penalty.
        # We add penalty proportional to count.
        if count > 0:
            penalty = penalty_scale * count
            perturbed[u, v] += penalty
            perturbed[v, u] += penalty
            
    return perturbed
