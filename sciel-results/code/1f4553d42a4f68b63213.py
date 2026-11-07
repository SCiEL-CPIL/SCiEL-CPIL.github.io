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
    - Uses soft-saturating function for smooth, bounded penalties.
    """
    t0 = time.time()
    n = edge_distance.shape[0]
    
    # Compute mean distance for scaling
    triu_idx = np.triu_indices(n, k=1)
    mean_dist = np.mean(edge_distance[triu_idx])
    if mean_dist < 1e-12:
        mean_dist = 1.0

    # Soft-saturating usage factor: u/(1+u) is smooth, bounded in [0,1), monotonic
    u = edge_n_used.astype(np.float64)
    sat = u / (1.0 + u)
    
    # Build the modified distance matrix (start with true distances)
    modified = edge_distance.copy()

    # Light additive global penalty on all edges based on usage
    # This gently discourages frequently-used edges without breaking the metric structure too much
    global_scale = 0.03 * mean_dist
    modified += global_scale * sat

    # Strong penalty on edges in the current tour to force 2-opt to remove them
    # Set to a large finite value to effectively exclude them from consideration
    # while keeping the matrix finite as required
    # The value should be large enough to discourage use but not cause numerical issues
    # 1e6 is large relative to typical TSP distances in [0, 1]
    INF_VAL = 1e6
    
    for i in range(n):
        j = local_opt_tour[i]
        modified[i, j] = INF_VAL
        modified[j, i] = INF_VAL

    return modified
# EVOLVE-BLOCK-END
