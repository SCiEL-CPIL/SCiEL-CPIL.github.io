"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty step for TSP.
    
    Strategy: rank-based soft constraint.
    1. Sort edges by usage count (edge_n_used).
    2. Assign a penalty that is a convex function (square) of the normalized rank.
       This creates a steep barrier against the most frequently used edges
       while leaving low-usage edges nearly unperturbed.
    3. Additionally, apply a direct penalty to the edges of the current local
       optimum tour to break the local minimum.
    
    Parameters:
    -----------
    edge_distance : np.ndarray
        True Euclidean distance matrix (n x n).
    local_opt_tour : list or np.ndarray
        Current tour as a list of node indices.
    edge_n_used : np.ndarray
        Symmetric count matrix of edge usage (n x n).
    
    Returns:
    --------
    np.ndarray
        Modified distance matrix with penalties applied.
    """
    start_time = time.time()
    deadline = start_time + 5.0  # Safety margin: must finish well before 330s total
    
    n = edge_distance.shape[0]
    
    # Get the current tour edges
    tour = list(local_opt_tour)
    tour_edges = set()
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        if u != v:
            tour_edges.add((min(u, v), max(u, v)))
    
    # Build a matrix of tour edge indicators
    tour_indicator = np.zeros_like(edge_distance)
    for (u, v) in tour_edges:
        tour_indicator[u, v] = 1.0
        tour_indicator[v, u] = 1.0
    
    # Compute rank-based penalty on edge_n_used
    # Flatten edge_n_used to get all unique edge usages
    # Since edge_n_used is symmetric, we only need the upper triangle
    upper_idx = np.triu_indices(n, k=1)
    edge_usages = edge_n_used[upper_idx]
    
    # Sort edges by usage count
    sorted_usages = np.sort(edge_usages)
    num_edges = len(sorted_usages)
    
    # Create a mapping from usage value to rank
    # For each edge, compute its rank among all edges
    # We want: high usage -> high rank -> high penalty
    # Use the rank of each edge's usage count
    
    # Create a rank matrix: for each edge, what is its usage rank?
    # Since we need to handle ties, we use average rank or just sort and assign
    rank_matrix = np.zeros_like(edge_distance)
    
    if num_edges > 0:
        # Get the sorted order
        sort_indices = np.argsort(sorted_usages)
        ranks = np.zeros(num_edges)
        # Assign ranks: 0 to num_edges-1 based on sorted position
        # But we want the rank of the original position
        # So: ranks[sort_indices] = arange(num_edges)
        ranks[sort_indices] = np.arange(num_edges)
        
        # Normalize ranks to [0, 1]
        if num_edges > 1:
            normalized_ranks = ranks / (num_edges - 1)
        else:
            normalized_ranks = np.zeros(1)
        
        # Apply convex penalty: square of normalized rank
        # This creates a steep barrier for high-rank (high-usage) edges
        penalty_values = normalized_ranks ** 2
        
        # Map back to the matrix
        rank_matrix[upper_idx] = penalty_values
        rank_matrix[np.transpose(upper_idx)] = penalty_values
    
    # Scale the rank-based penalty
    # Use a scale factor based on the mean distance
    mean_dist = np.mean(edge_distance)
    
    # The penalty should be significant enough to disrupt the local optimum
    # but not so large that it completely dominates the true distances
    rank_penalty_scale = 0.5 * mean_dist
    
    # Direct penalty for current tour edges
    # These edges are part of the local minimum and should be strongly penalized
    tour_penalty_scale = 0.3 * mean_dist
    
    # Combine penalties
    # The rank-based penalty provides a smooth, gradient-like push
    # The tour penalty provides a direct kick to escape the local minimum
    penalized_distance = edge_distance.copy()
    penalized_distance += rank_penalty_scale * rank_matrix
    penalized_distance += tour_penalty_scale * tour_indicator
    
    # Ensure the result is finite and non-negative
    penalized_distance = np.maximum(penalized_distance, 0.0)
    penalized_distance[~np.isfinite(penalized_distance)] = edge_distance[~np.isfinite(penalized_distance)]
    
    # Check deadline
    if time.time() > deadline:
        # Return a safe fallback if we're running out of time
        return edge_distance
    
    return penalized_distance
# EVOLVE-BLOCK-END
