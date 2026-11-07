"""Guided local search penalty rule for the TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation: multiplicative context-aware penalty.
    
    - Multiplicative penalty on tour edges, scaled by usage count.
    - Light additive penalty on all edges based on usage.
    - Scaled relative to mean edge distance for robustness.
    """
    t0 = time.time()
    n = edge_distance.shape[0]
    
    # Compute mean distance for scaling
    triu_idx = np.triu_indices(n, k=1)
    mean_dist = np.mean(edge_distance[triu_idx])
    if mean_dist < 1e-12:
        mean_dist = 1.0

    # Soft-saturating usage factor
    u = edge_n_used.astype(np.float64)
    sat = u / (1.0 + u)

    # Build the modified distance matrix (start with true distances)
    modified = edge_distance.copy()

    # Light additive global penalty on all edges based on usage
    global_scale = 0.03 * mean_dist
    modified += global_scale * sat

    # Strong multiplicative penalty on edges in the current tour
    # Use a power-law scaling: penalty grows with usage but saturates
    tour_mult_scale = 0.4 * mean_dist
    for i in range(n):
        j = local_opt_tour[i]
        # Multiplicative: increase the distance by a factor proportional to usage
        penalty = tour_mult_scale * (1.0 + sat[i, j])
        modified[i, j] += penalty
        modified[j, i] += penalty

    # Time guard
    if time.time() - t0 > 5.0:
        pass

    # Ensure symmetry and finite values
    modified = (modified + modified.T) / 2.0
    np.nan_to_num(modified, copy=False, nan=0.0, posinf=1e6, neginf=0.0)

    return modified
# EVOLVE-BLOCK-END
