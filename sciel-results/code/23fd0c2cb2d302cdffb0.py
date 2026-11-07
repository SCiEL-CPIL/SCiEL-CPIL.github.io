"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step of guided local search for TSP.
    
    Uses a repulsive force model where penalty is inversely proportional to edge length,
    so short high-value edges receive disproportionately large penalties.
    Combined with saturating usage penalty and immediate tour-edge kick.
    """
    n = edge_distance.shape[0]
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Build penalty from cumulative usage with saturating normalization
    max_used = edge_n_used.max()
    if max_used > 0:
        # Saturating penalty: grows with usage but caps out
        alpha = 0.5
        usage_penalty = alpha * edge_n_used / (1.0 + edge_n_used)
    else:
        usage_penalty = np.zeros_like(edge_n_used)
    
    # Add immediate penalty to edges in the current tour
    tour_penalty = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = local_opt_tour[i]
        k = local_opt_tour[(i + 1) % n]
        tour_penalty[i, j] += 1.0
        tour_penalty[j, i] += 1.0
    
    # Geometry-aware repulsive weighting: inverse distance penalty
    # Short edges get larger relative penalty, forcing exploration of alternatives
    # Use a smoothed inverse to avoid division by zero for very short edges
    min_dist = np.min(edge_distance[np.triu_indices(n, k=1)])
    if min_dist > 0:
        # Inverse distance weighting with smoothing
        # weight = 1 / (edge_distance + epsilon) normalized
        eps = 0.01 * min_dist
        inv_dist = 1.0 / (edge_distance + eps)
        # Normalize so that the mean of inv_dist over upper triangle is 1
        upper_idx = np.triu_indices(n, k=1)
        mean_inv = np.mean(inv_dist[upper_idx])
        if mean_inv > 0:
            inv_dist_norm = inv_dist / mean_inv
        else:
            inv_dist_norm = np.ones_like(edge_distance)
    else:
        inv_dist_norm = np.ones_like(edge_distance)
    
    # Combine penalties: usage penalty weighted by inverse distance + tour kick
    # The inverse distance weighting makes short edges more heavily penalized
    gamma = 0.4  # Weight for the geometry-aware component
    beta = 0.3   # Weight for immediate tour penalty
    
    # Usage penalty is also weighted by inverse distance for geometry-awareness
    geometry_weighted_usage = usage_penalty * inv_dist_norm
    geometry_weighted_tour = tour_penalty * inv_dist_norm
    
    total_penalty = gamma * geometry_weighted_usage + beta * geometry_weighted_tour
    
    # Scale the penalty to be comparable to typical edge lengths
    mean_dist = np.mean(edge_distance[np.triu_indices(n, k=1)])
    penalty_scale = 0.15 * mean_dist
    
    perturbed = edge_distance + penalty_scale * total_penalty
    
    # Safety: ensure finite values
    perturbed = np.where(np.isfinite(perturbed), perturbed, edge_distance)
    
    # Deadline check - if we're close to the limit, return immediately
    if time.time() > deadline:
        return perturbed
    
    return perturbed
# EVOLVE-BLOCK-END
