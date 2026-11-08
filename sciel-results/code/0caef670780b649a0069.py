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
    
    # Get true distances for current tour edges
    u = tour
    v = np.roll(tour, -1)
    
    current_edge_lengths = edge_distance[u, v]
    tour_len = np.sum(current_edge_lengths)
    
    # If the tour is degenerate or something weird, fallback
    if not np.isfinite(tour_len) or tour_len == 0:
        tour_len = np.sum(edge_distance) / 2.0  # rough estimate
        if not np.isfinite(tour_len) or tour_len == 0:
            tour_len = 1.0  # Fallback for unit square 100 pts
    
    # Adaptive penalty: scale so that the penalty on the current tour's edges
    # makes the tour's penalized length exceed its true length by a margin.
    # This ensures 2-opt will try to escape the current local optimum.
    
    # Sum of edge_n_used on current tour edges
    tour_edge_usage_sum = np.sum(edge_n_used[u, v])
    
    if tour_edge_usage_sum > 0:
        # We want: sum over tour edges of (penalty_coeff * edge_n_used[i,j]) >= margin
        # So penalty_coeff = margin / tour_edge_usage_sum
        # Choose margin as a fraction of tour_len to be scale-invariant
        margin = 0.1 * tour_len
        penalty_coeff = margin / tour_edge_usage_sum
    else:
        # No usage yet, use a small default penalty
        avg_edge_len = tour_len / n
        penalty_coeff = 0.01 * avg_edge_len
    
    # Create the penalized distance matrix
    penalized_dist = edge_distance + penalty_coeff * edge_n_used
    
    # Ensure the diagonal is 0
    np.fill_diagonal(penalized_dist, 0.0)
    
    # Ensure symmetry
    penalized_dist = (penalized_dist + penalized_dist.T) / 2.0
    
    # Check for finiteness
    if not np.all(np.isfinite(penalized_dist)):
        penalized_dist = np.nan_to_num(penalized_dist, nan=0.0, posinf=np.max(edge_distance), neginf=0.0)
    
    return penalized_dist
