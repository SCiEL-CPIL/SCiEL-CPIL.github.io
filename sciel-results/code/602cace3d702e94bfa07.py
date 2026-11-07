"""Guided local search penalty rule for the traveling salesman problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Refinement of the distance-aware penalty design:
    1. Use a soft-min based adaptive penalty that focuses on edges competing
       with the current tour's structure.
    2. Apply a smooth, bounded usage penalty with adaptive scaling based on
       the current usage distribution.
    3. Directly penalize tour edges with a decaying strength to encourage
       escape from the local optimum.
    4. All penalties are applied multiplicatively to maintain positive, finite
       distances.
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
    tour_mean = np.mean(tour_edges)
    tour_std = np.std(tour_edges)
    if tour_std < 1e-10:
        tour_std = tour_median * 0.1 + 1e-10
    
    # Compute the median of all edges for normalization
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    upper_edges = edge_distance[mask]
    global_median = np.median(upper_edges)
    if global_median <= 0:
        global_median = 1.0
    
    # 1. Distance-aware penalty using a soft-min surrogate:
    #    Penalize edges that are similar in length to the current tour's edges.
    #    Use a Gaussian kernel centered on the tour mean, with width adapted
    #    to the tour's edge length distribution.
    z_score = (edge_distance - tour_mean) / tour_std
    
    # Adaptive sigma: narrower when tour edges are tightly distributed
    sigma = 1.2 + 0.3 * (tour_std / (global_median + 1e-10))
    sigma = np.clip(sigma, 0.8, 2.5)
    
    gaussian_weight = np.exp(-0.5 * (z_score / sigma) ** 2)
    
    # Scale: stronger penalty for edges near the tour mean length
    alpha_distance = 0.30
    distance_penalty = alpha_distance * gaussian_weight
    
    # 2. Usage-based penalty with smooth saturation and adaptive scaling:
    #    The penalty grows with usage but saturates to avoid extreme values.
    #    Scale adaptively based on the current usage distribution.
    max_usage = np.max(edge_n_used)
    if max_usage > 0:
        usage_scale = 1.0 / (1.0 + 0.1 * max_usage)
    else:
        usage_scale = 1.0
    
    alpha_usage = 0.22
    clipped_usage = np.clip(edge_n_used, 0, 200)
    # Smooth saturation: x / (1 + x) is bounded and smooth
    usage_penalty = alpha_usage * usage_scale * (clipped_usage / (1.0 + clipped_usage))
    
    # 3. Tour-edge penalty: directly penalize edges in the current tour
    #    with a strength that decays slightly with usage to avoid over-penalizing
    #    edges that have already been heavily penalized.
    tour_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_mask[a, b] = 1.0
        tour_mask[b, a] = 1.0
    
    # Decay factor: reduce immediate penalty if these edges are already heavily used
    tour_edge_usage = np.array([edge_n_used[local_opt_tour[i], local_opt_tour[(i + 1) % n]] 
                                 for i in range(n)])
    mean_tour_usage = np.mean(tour_edge_usage) if len(tour_edge_usage) > 0 else 0
    decay_factor = 1.0 / (1.0 + 0.05 * mean_tour_usage)
    
    immediate_scale = 0.25 * decay_factor
    immediate_penalty = immediate_scale * tour_mask
    
    # Check deadline
    if time.time() > deadline:
        result = edge_distance * (1.0 + immediate_penalty)
        result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=0.0)
        return result
    
    # 4. Combine all penalties
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
