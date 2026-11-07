"""Guided local search penalty rule for the traveling salesman problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Structural change: Instead of a uniform repulsive force on all edges,
    we use a distance-aware penalty that selectively penalizes edges based
    on their rank relative to the current tour's edge length distribution.
    
    Key insights:
    1. Edges that are similar in length to the current tour's edges are
       penalized more heavily (they are "competitors" for the same role).
    2. Very short edges get a moderate penalty (they are good but may be
       part of the current suboptimal structure).
    3. Very long edges get a smaller penalty (they are unlikely to be useful
       anyway, so we don't need to push 2-opt away from them).
    4. Usage penalty is applied with a logistic saturation to avoid extreme
       values while still providing strong pressure on frequently-used edges.
    """
    start_time = time.time()
    n = edge_distance.shape[0]
    deadline = start_time + 280.0
    
    # Get the current tour's edge lengths
    tour_edges = []
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_edges.append(edge_distance[a, b])
    tour_edges = np.array(tour_edges)
    
    # Compute statistics of the tour edge lengths
    tour_median = np.median(tour_edges)
    tour_std = np.std(tour_edges)
    if tour_std < 1e-10:
        tour_std = tour_median * 0.1 + 1e-10
    
    # Compute the median of all edges for normalization
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    upper_edges = edge_distance[mask]
    global_median = np.median(upper_edges)
    if global_median <= 0:
        global_median = 1.0
    
    # 1. Distance-aware penalty: penalize edges that are close in length to
    #    the current tour's edges. This creates a "forbidden zone" around the
    #    typical edge lengths of the current solution.
    #    Use a Gaussian-like kernel centered on the tour median.
    normalized_dist = edge_distance / global_median
    # z-score relative to tour edge distribution
    z_score = (edge_distance - tour_median) / tour_std
    
    # Gaussian penalty: peaks at z=0 (edges similar to tour edges)
    # Width controls how selective the penalty is
    sigma = 1.5
    gaussian_weight = np.exp(-0.5 * (z_score / sigma) ** 2)
    
    # Scale the penalty: stronger for edges near the tour median length
    alpha_distance = 0.35
    distance_penalty = alpha_distance * gaussian_weight
    
    # 2. Usage-based penalty with logistic saturation
    # This provides a smooth, bounded penalty that grows with usage
    alpha_usage = 0.25
    clipped_usage = np.clip(edge_n_used, 0, 100)
    # Logistic function: saturates at 1.0 for large usage values
    usage_penalty = alpha_usage * (clipped_usage / (1.0 + clipped_usage))
    
    # 3. Tour-edge penalty: directly penalize edges in the current tour
    #    to force 2-opt to break away
    immediate_scale = 0.20
    tour_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_mask[a, b] = 1.0
        tour_mask[b, a] = 1.0
    
    immediate_penalty = immediate_scale * tour_mask
    
    # Check deadline
    if time.time() > deadline:
        result = edge_distance * (1.0 + immediate_penalty)
        result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=0.0)
        return result
    
    # 4. Combine all penalties
    #    The distance-aware penalty and usage penalty are additive,
    #    and the tour-edge penalty is applied on top.
    total_penalty = distance_penalty + usage_penalty + immediate_penalty
    
    # Apply multiplicatively to keep all distances positive and finite
    result = edge_distance * (1.0 + total_penalty)
    
    # Ensure finite values
    result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=0.0)
    
    # Final deadline check
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
