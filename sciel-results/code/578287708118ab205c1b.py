"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search on TSP.
    
    Combines:
    1. Usage-based penalty (edges used more often get higher penalty)
    2. Topological disruption: heavily penalize the shortest edges in the current tour
       to force 2-opt to find structurally different tours
    
    Args:
        edge_distance: (N, N) array of true Euclidean distances.
        local_opt_tour: (N,) array of node indices forming the current tour.
        edge_n_used: (N, N) array of symmetric counts of edge usage.
        
    Returns:
        (N, N) array of perturbed distances (finite, same shape).
    """
    n = len(local_opt_tour)
    if n < 2:
        return edge_distance.copy()
    
    # Ensure we work with float
    perturbed = edge_distance.astype(np.float64).copy()
    
    # Get the edges of the current tour
    tour = local_opt_tour
    tour_edges = []
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        tour_edges.append((u, v))
    
    # Compute usage-based penalty
    # Normalize edge_n_used to [0, 1] range
    max_used = edge_n_used.max()
    if max_used > 0:
        norm_used = edge_n_used / max_used
    else:
        norm_used = np.zeros_like(edge_n_used)
    
    # Base penalty: proportional to usage
    # Use a nonlinear scaling to emphasize frequently used edges
    penalty_base = 0.5 * (norm_used ** 0.5)  # sqrt to spread out
    
    # Topological disruption: identify the shortest k edges in the current tour
    # and give them extra heavy penalty to force structural change
    # k adapts based on how many times edges have been used
    if len(tour_edges) > 0:
        edge_lengths = []
        for (u, v) in tour_edges:
            edge_lengths.append((edge_distance[u, v], u, v))
        
        # Sort by length
        edge_lengths.sort(key=lambda x: x[0])
        
        # Adaptive k: more disruption when edges are heavily used
        total_usage = sum(edge_n_used[u, v] for (u, v) in tour_edges)
        avg_usage = total_usage / len(tour_edges) if len(tour_edges) > 0 else 0
        k = max(1, min(n // 4, int(1 + 2 * avg_usage)))
        
        # Apply heavy penalty to the k shortest tour edges
        for i in range(k):
            _, u, v = edge_lengths[i]
            # Strong penalty: multiply by a large factor
            perturbed[u, v] *= (1.0 + 2.0 + avg_usage)
            perturbed[v, u] *= (1.0 + 2.0 + avg_usage)
    
    # Apply the base usage penalty to all edges
    # Only penalize, don't reduce
    perturbed = perturbed * (1.0 + penalty_base)
    
    # Ensure symmetry
    perturbed = (perturbed + perturbed.T) / 2.0
    
    # Ensure all values are finite and positive
    perturbed = np.where(np.isfinite(perturbed), perturbed, 1e10)
    perturbed = np.maximum(perturbed, 1e-10)
    
    return perturbed
# EVOLVE-BLOCK-END
