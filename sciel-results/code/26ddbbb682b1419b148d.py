"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search.
    
    Uses a combined strategy:
    1. Penalize edges that are part of the current local optimum tour.
    2. Scale the penalty by the number of times each edge has been used.
    3. Add a small random noise to break symmetry and encourage exploration.
    
    Args:
        edge_distance: (N, N) array of true Euclidean distances.
        local_opt_tour: (N,) array of node indices forming the current tour.
        edge_n_used: (N, N) array of symmetric counts of edge usage.
        
    Returns:
        (N, N) array of perturbed distances.
    """
    n = len(local_opt_tour)
    
    # Create a copy to avoid modifying the input
    perturbed = edge_distance.copy()
    
    # Build a set of edges in the current tour for quick lookup
    tour_edges = set()
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges.add((min(u, v), max(u, v)))
    
    # Calculate statistics for adaptive penalty
    # Average usage count of tour edges
    tour_usage = []
    for (u, v) in tour_edges:
        tour_usage.append(edge_n_used[u, v])
    
    if len(tour_usage) > 0:
        avg_usage = np.mean(tour_usage)
        max_usage = np.max(tour_usage)
    else:
        avg_usage = 0
        max_usage = 1
    
    # Adaptive penalty factor based on usage
    # Higher usage -> higher penalty to encourage moving away
    base_penalty = 0.5
    
    # Scale penalty by normalized usage
    if max_usage > 0:
        usage_scale = 1.0 + (avg_usage / (max_usage + 1.0))
    else:
        usage_scale = 1.0
    
    # Apply penalty to tour edges
    for (u, v) in tour_edges:
        # Penalty proportional to how many times this edge has been used
        penalty = base_penalty * usage_scale * (1.0 + edge_n_used[u, v] / (max_usage + 1.0))
        perturbed[u, v] += penalty
        perturbed[v, u] += penalty
    
    # Add small random noise to all off-diagonal elements to break ties
    # and encourage exploration
    noise_scale = 0.01 * np.std(edge_distance[np.triu_indices(n, k=1)])
    mask = ~np.eye(n, dtype=bool)
    noise = np.random.normal(0, noise_scale, size=(n, n))
    noise = (noise + noise.T) / 2.0  # Make symmetric
    noise[np.eye(n, dtype=bool)] = 0.0
    perturbed += noise
    
    # Ensure diagonal is zero
    np.fill_diagonal(perturbed, 0.0)
    
    # Ensure symmetry
    perturbed = (perturbed + perturbed.T) / 2.0
    
    # Ensure finite values
    perturbed = np.where(np.isfinite(perturbed), perturbed, edge_distance)
    
    return perturbed
# EVOLVE-BLOCK-END
