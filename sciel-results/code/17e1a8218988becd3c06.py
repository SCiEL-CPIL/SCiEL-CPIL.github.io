"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Targeted Escape Strategy for Guided Local Search.
    
    Instead of penalizing all edges based on historical usage (which distorts the
    global metric), this strategy applies a strong, adaptive penalty ONLY to the
    edges of the current local optimum tour. This forces the subsequent 2-opt
    to break the current solution structure, while leaving the rest of the metric
    space undistorted to facilitate finding a new, potentially better basin.
    """
    n = len(local_opt_tour)
    if n <= 2:
        return edge_distance.copy()

    # Create a copy of the true distance matrix
    perturbed_dist = edge_distance.copy()
    
    # Identify edges in the current tour
    # local_opt_tour is a permutation of 0..n-1 representing the tour order
    # Edges are (tour[i], tour[(i+1)%n])
    
    # We want to penalize these edges.
    # The penalty magnitude should be strong enough to force 2-opt to remove them.
    # A multiplier of 5.0 to 10.0 is typically sufficient for 2-opt to break the cycle.
    # We can make it adaptive: if an edge has been used many times, it's likely a "trap" edge.
    
    max_usage = edge_n_used.max()
    if max_usage == 0:
        base_penalty = 5.0
    else:
        # If usage is high, we are stuck. Increase penalty.
        # Normalize usage of current edges relative to max
        current_edge_usages = []
        for i in range(n):
            u = local_opt_tour[i]
            v = local_opt_tour[(i + 1) % n]
            current_edge_usages.append(edge_n_used[u, v])
        
        avg_usage = np.mean(current_edge_usages)
        # Scale penalty: base 5.0, up to 15.0 if avg usage is max
        base_penalty = 5.0 + 10.0 * (avg_usage / max_usage)
        
    # Apply the penalty to the current tour's edges
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        # Multiply the distance by the penalty factor
        # This makes these edges very expensive in the perturbed metric
        perturbed_dist[u, v] *= base_penalty
        perturbed_dist[v, u] *= base_penalty

    return perturbed_dist
# EVOLVE-BLOCK-END
