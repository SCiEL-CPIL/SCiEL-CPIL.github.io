"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Args:
        edge_distance: true Euclidean distance matrix (n x n)
        local_opt_tour: current tour as a sequence of node indices
        edge_n_used: symmetric count matrix of how often each edge has been used
    
    Returns:
        Perturbed distance matrix with penalties applied to frequently used edges.
    """
    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()
    
    # Extract edges in the current local optimal tour
    tour = list(local_opt_tour)
    tour_edges = set()
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        tour_edges.add((min(u, v), max(u, v)))
    
    # Create a penalty matrix
    # Strategy: penalize edges that are in the current tour AND have been used frequently
    # Use a smooth, adaptive penalty based on usage frequency
    
    max_used = edge_n_used.max()
    if max_used == 0:
        max_used = 1
    
    # Normalize usage counts
    normalized_usage = edge_n_used / max_used
    
    # Penalty for each edge:
    # - Base penalty proportional to how many times the edge has been used
    # - Additional penalty for edges in the current local optimum
    # - Use a soft penalty that scales with usage frequency
    
    # Create the perturbed distance matrix
    perturbed = edge_distance.copy()
    
    # For each edge in the tour, apply a penalty
    # The penalty should be strong enough to break the local optimum but not so strong
    # that it destroys useful structure
    
    # Adaptive penalty: stronger for more frequently used edges
    # Use a smooth function: penalty = base * (1 + alpha * normalized_usage)
    # where alpha controls the strength of the penalty
    
    alpha = 0.5  # Penalty strength parameter
    
    for (u, v) in tour_edges:
        usage_count = edge_n_used[u, v]
        norm_usage = usage_count / max_used
        
        # Penalty increases with usage, with a smooth transition
        # Use a hyperbolic-like function for smoothness
        penalty_factor = 1.0 + alpha * norm_usage
        
        # Apply penalty to both directions (symmetric)
        perturbed[u, v] *= penalty_factor
        perturbed[v, u] *= penalty_factor
    
    # Additionally, add a small penalty to ALL edges that have been used,
    # not just those in the current tour, to encourage exploration
    # This helps escape from broader local optima
    
    global_alpha = 0.1  # Smaller penalty for non-tour edges
    
    # Vectorized version for non-tour edges
    # Create a mask of edges not in the tour
    mask = np.ones((n, n), dtype=bool)
    for (u, v) in tour_edges:
        mask[u, v] = False
        mask[v, u] = False
    
    # Apply global penalty to non-tour edges based on their usage
    # Only apply where usage > 0 and not in current tour
    usage_mask = (edge_n_used > 0) & mask
    if np.any(usage_mask):
        # Smooth penalty: 1 + global_alpha * normalized_usage
        penalty = 1.0 + global_alpha * normalized_usage
        perturbed[usage_mask] *= penalty[usage_mask]
    
    # Ensure the matrix remains symmetric
    perturbed = (perturbed + perturbed.T) / 2.0
    
    # Ensure diagonal is 0 (distance from node to itself)
    np.fill_diagonal(perturbed, 0.0)
    
    # Ensure all values are finite
    perturbed = np.where(np.isfinite(perturbed), perturbed, edge_distance)
    
    return perturbed
# EVOLVE-BLOCK-END
