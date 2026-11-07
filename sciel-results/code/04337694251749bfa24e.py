"""Guided local search penalty rule for TSP - rewritten."""
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
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    n = edge_distance.shape[0]
    result = edge_distance.astype(np.float64).copy()
    
    # Get edges in the current local optimum tour
    tour_edges = []
    for i in range(len(local_opt_tour)):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % len(local_opt_tour)]
        tour_edges.append((u, v))
    
    # Normalize edge usage counts
    max_used = edge_n_used.max()
    if max_used > 0:
        # Scale penalty by usage: more used edges get more penalty
        # Use a softer scaling to avoid extreme values
        usage_scale = edge_n_used / (1.0 + max_used)
    else:
        usage_scale = np.zeros_like(edge_n_used)
    
    # Strategy 1: Penalize all edges based on usage (like seed but stronger)
    # Penalty factor: 1 + alpha * usage_normalized
    alpha = 0.5  # Stronger than seed's 0.1
    penalty_all = 1.0 + alpha * usage_scale
    
    # Strategy 2: Extra penalty for edges in the current tour
    # These are the edges we want to escape from
    tour_penalty_mask = np.zeros_like(edge_n_used)
    for u, v in tour_edges:
        tour_penalty_mask[u, v] = 1.0
        tour_penalty_mask[v, u] = 1.0
    
    # Extra multiplier for tour edges: 1 + beta * usage
    beta = 1.0
    tour_extra = 1.0 + beta * usage_scale * tour_penalty_mask
    
    # Combine: apply penalty to all edges, extra to tour edges
    combined_penalty = penalty_all * tour_extra
    
    # Apply the penalty
    result = result * combined_penalty
    
    # Ensure symmetry
    result = (result + result.T) / 2.0
    
    # Ensure diagonal is zero
    np.fill_diagonal(result, 0.0)
    
    # Ensure all values are finite and positive
    result = np.where(np.isfinite(result) & (result > 0), result, edge_distance)
    np.fill_diagonal(result, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        # Return the best result we have
        pass
    
    return result
# EVOLVE-BLOCK-END
