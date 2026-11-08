import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Updates the edge distance matrix by adding penalties to edges
    frequently used in previous local optima (Guided Local Search).
    
    Uses an adaptive penalty strategy where the penalty magnitude is scaled
    to make the current local optimum tour sub-optimal under the penalized
    distances, forcing 2-opt to escape the current basin.
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
    
    # Adaptive GLS penalty:
    # We want to penalize edges in the current tour enough that the tour becomes
    # non-optimal under the penalized distances. The key insight is that if we add
    # a penalty to each edge in the current tour, the total increase in tour length
    # under penalized distances should be significant enough that 2-opt finds a
    # different tour.
    #
    # Strategy: scale penalty so that the sum of penalties on current tour edges
    # is a meaningful fraction of the tour length.
    # 
    # Let penalty_coeff * sum(edge_n_used[i,j] for (i,j) in tour) = margin * tour_len
    # This gives: penalty_coeff = margin * tour_len / sum(edge_n_used[i,j] for (i,j) in tour)
    #
    # But we also want to penalize edges used in previous tours, not just current.
    # So we apply the penalty to all edges based on edge_n_used.
    #
    # The margin parameter controls how strongly we want to escape. A margin of 0.1
    # means the penalized tour length increases by 10% of the true tour length.
    
    # Compute the sum of usage counts on the current tour edges
    tour_usage_sum = np.sum(edge_n_used[u, v])
    
    if tour_usage_sum > 0 and np.isfinite(tour_usage_sum):
        # Adaptive penalty coefficient
        margin = 0.15  # 15% increase in tour length due to penalties
        penalty_coeff = margin * tour_len / tour_usage_sum
    else:
        # Fallback: use a fixed fraction of average edge length
        penalty_coeff = 0.1 * avg_edge_len
    
    # Ensure penalty_coeff is reasonable
    if not np.isfinite(penalty_coeff) or penalty_coeff < 0:
        penalty_coeff = 0.1 * avg_edge_len
    
    # Create the penalized distance matrix
    # Add penalty proportional to usage count
    penalized_dist = edge_distance + penalty_coeff * edge_n_used
    
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
