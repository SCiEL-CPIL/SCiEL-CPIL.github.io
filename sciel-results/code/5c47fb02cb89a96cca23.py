import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Updates the edge distance matrix by adding penalties to edges
    frequently used in previous local optima (Guided Local Search).
    
    Combines adaptive global penalty scaling (from current program) with
    targeted sparse penalties on the current tour edges (from inspiration 1).
    """
    start_time = time.time()
    deadline = start_time + 2.0  # Safety margin
    
    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()
    
    tour = np.asarray(local_opt_tour)
    
    # Extract edges of the current local optimum tour
    u = tour
    v = np.roll(tour, -1)
    
    # Get true distances for these edges
    current_edge_lengths = edge_distance[u, v]
    tour_len = np.sum(current_edge_lengths)
    avg_edge_len = np.mean(current_edge_lengths)
    
    # Fallbacks for degenerate cases
    if not np.isfinite(tour_len) or tour_len == 0:
        tour_len = np.sum(edge_distance) / n
        if not np.isfinite(tour_len) or tour_len == 0:
            tour_len = 1.0  # Fallback for unit square 100 pts
    
    if not np.isfinite(avg_edge_len) or avg_edge_len == 0:
        avg_edge_len = np.mean(edge_distance)
        if not np.isfinite(avg_edge_len) or avg_edge_len == 0:
            avg_edge_len = 0.1
    
    # Create a copy of the true distance matrix
    penalized_dist = edge_distance.copy()
    
    # Compute the sum of usage counts on the current tour edges
    tour_usage_sum = np.sum(edge_n_used[u, v])
    
    # Adaptive global penalty coefficient
    if tour_usage_sum > 0 and np.isfinite(tour_usage_sum):
        # Scale so that total penalty on current tour is a fraction of tour length
        margin = 0.12
        penalty_coeff = margin * tour_len / tour_usage_sum
    else:
        # Fallback: use a fixed fraction of average edge length
        penalty_coeff = 0.08 * avg_edge_len
    
    # Ensure penalty_coeff is reasonable
    if not np.isfinite(penalty_coeff) or penalty_coeff < 0:
        penalty_coeff = 0.08 * avg_edge_len
    
    # Apply global penalty based on accumulated usage
    penalized_dist += penalty_coeff * edge_n_used
    
    # Add targeted sparse penalty to the current tour edges
    # This makes the current local optimum noticeably more expensive,
    # forcing 2-opt to find a different tour
    targeted_penalty = 0.4 * avg_edge_len
    penalized_dist[u, v] += targeted_penalty
    penalized_dist[v, u] += targeted_penalty
    
    # Ensure the diagonal is 0
    np.fill_diagonal(penalized_dist, 0.0)
    
    # Ensure symmetry
    penalized_dist = (penalized_dist + penalized_dist.T) / 2.0
    
    # Check for finiteness
    if not np.all(np.isfinite(penalized_dist)):
        penalized_dist = np.nan_to_num(penalized_dist, nan=0.0, posinf=np.max(edge_distance) * 2, neginf=0.0)
    
    # Time check - if we're close to deadline, return immediately
    if time.time() > deadline - 0.1:
        return penalized_dist
    
    return penalized_dist
