# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Topological disruption strategy for guided local search.
    
    Instead of scalar penalties, we identify the most frequently used edges
    in the current tour and set their distances to a large value, forcing
    2-opt to find alternative connections. We also apply a mild penalty to
    other tour edges based on usage frequency.
    """
    n = edge_distance.shape[0]
    
    # Work on a copy to avoid modifying the input
    perturbed = edge_distance.copy()
    
    # Extract the edges of the current tour
    # local_opt_tour is a permutation of [0, ..., n-1] representing the tour
    # The tour edges are (tour[i], tour[i+1]) for i in range(n), with wrap-around
    tour_edges = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        # Ensure consistent ordering for the symmetric matrix
        if u > v:
            u, v = v, u
        tour_edges.append((u, v))
    
    # Count usage of each edge in the current tour from edge_n_used
    # edge_n_used is a symmetric count matrix
    edge_usage = np.zeros((n, n), dtype=np.float64)
    for u, v in tour_edges:
        edge_usage[u, v] = edge_n_used[u, v]
        edge_usage[v, u] = edge_n_used[u, v]
    
    # Strategy: 
    # 1. Identify the k most frequently used tour edges and set them to a large value
    # 2. Apply a graduated penalty to remaining tour edges based on usage
    
    # Determine how many edges to disrupt (topological disruption)
    # Use a fraction of the tour edges, e.g., the top 20% most used
    k_disrupt = max(1, int(0.2 * n))
    
    # Get the usage values for tour edges
    tour_edge_usages = []
    for u, v in tour_edges:
        tour_edge_usages.append((edge_n_used[u, v], u, v))
    
    # Sort by usage in descending order
    tour_edge_usages.sort(key=lambda x: x[0], reverse=True)
    
    # Set the top k most used edges to a large value
    large_value = 1e6
    
    # First, apply a mild penalty to all tour edges based on usage
    max_usage = edge_n_used.max()
    if max_usage > 0:
        for u, v in tour_edges:
            usage = edge_n_used[u, v]
            # Mild penalty: scale by usage frequency
            penalty_factor = 1.0 + 0.5 * (usage / max_usage)
            perturbed[u, v] *= penalty_factor
            perturbed[v, u] *= penalty_factor
    
    # Now disrupt the top k most used edges by setting them to a large value
    for i in range(k_disrupt):
        usage, u, v = tour_edge_usages[i]
        perturbed[u, v] = large_value
        perturbed[v, u] = large_value
    
    # Ensure the matrix is symmetric
    perturbed = (perturbed + perturbed.T) / 2.0
    
    # Ensure no NaN or Inf values (large_value is finite)
    perturbed = np.nan_to_num(perturbed, nan=large_value, posinf=large_value, neginf=0.0)
    
    return perturbed
# EVOLVE-BLOCK-END
