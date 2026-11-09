# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Topological disruption strategy: Identify the shortest edges in the current
    tour and remove them (set to large penalty) to force restructuring.
    """
    n = edge_distance.shape[0]
    
    # Work on a copy to avoid modifying the input
    perturbed = edge_distance.copy()
    
    # Extract the edges of the current tour
    tour_edges = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        if u > v:
            u, v = v, u
        tour_edges.append((u, v))
        
    # Get distances of these tour edges from the true distance matrix
    edge_data = []
    for u, v in tour_edges:
        dist = edge_distance[u, v]
        edge_data.append((dist, u, v))
        
    # Sort by distance in ascending order to find the shortest edges
    edge_data.sort(key=lambda x: x[0])
    
    # Select the k shortest edges to disrupt
    # Disrupting a small fraction (5%) forces the tour to use alternative,
    # likely longer, connections, breaking the local greedy structure.
    k_disrupt = max(1, int(0.05 * n))
    
    # Set these edges to a large penalty value
    # Max distance in unit square is ~1.414. 100.0 is effectively infinity.
    large_penalty = 100.0
    
    for i in range(k_disrupt):
        _, u, v = edge_data[i]
        perturbed[u, v] = large_penalty
        perturbed[v, u] = large_penalty
        
    # Ensure the matrix is symmetric and finite
    perturbed = (perturbed + perturbed.T) / 2.0
    perturbed = np.nan_to_num(perturbed, nan=large_penalty, posinf=large_penalty, neginf=0.0)
    
    return perturbed
# EVOLVE-BLOCK-END
