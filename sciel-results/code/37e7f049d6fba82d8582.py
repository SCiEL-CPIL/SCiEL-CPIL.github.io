"""Guided local search penalty for TSP."""
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Return a modified distance matrix that penalizes edges used in the current
    local-optimal tour and historically used edges, guiding 2-opt away from
    the current basin.
    
    Strategy:
    - Penalize edges in the current tour with a strong, usage-dependent penalty.
    - Add a mild global penalty for historically used edges.
    - Use a smooth (logistic-like) scaling to avoid discontinuities.
    - Ensure the output is finite and same shape.
    """
    n = edge_distance.shape[0]
    
    # Work on a copy
    penalized = edge_distance.astype(np.float64).copy()
    
    # Identify edges in the current tour
    tour_edges = set()
    for i in range(len(local_opt_tour)):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % len(local_opt_tour)]
        if u != v:
            tour_edges.add((min(u, v), max(u, v)))
    
    # Compute a reference scale: median of non-zero distances
    mask = (edge_distance > 0) & (edge_distance < np.inf)
    if np.any(mask):
        ref_scale = np.median(edge_distance[mask])
    else:
        ref_scale = 1.0
    
    # Penalty for edges in current tour: strong, proportional to usage
    # Use a smooth function: penalty = ref_scale * alpha * (1 + beta * usage)
    # where alpha controls base strength, beta controls usage sensitivity
    alpha = 0.3
    beta = 0.5
    
    for (u, v) in tour_edges:
        usage = edge_n_used[u, v]
        # Smooth penalty: grows with usage but saturates somewhat
        penalty = ref_scale * alpha * (1.0 + beta * np.log1p(usage))
        penalized[u, v] += penalty
        penalized[v, u] += penalty
    
    # Global mild penalty for all historically used edges
    # Use a small additive penalty proportional to usage count
    global_alpha = 0.05
    penalized += global_alpha * ref_scale * edge_n_used
    
    # Ensure symmetry
    penalized = 0.5 * (penalized + penalized.T)
    
    # Ensure diagonal is zero
    np.fill_diagonal(penalized, 0.0)
    
    # Ensure finiteness
    penalized = np.where(np.isfinite(penalized), penalized, np.inf)
    # Replace any inf with a large finite value
    max_val = np.nanmax(penalized[~np.isinf(penalized)]) if np.any(~np.isinf(penalized)) else 1.0
    large_val = max_val * 1e6
    penalized = np.where(np.isinf(penalized), large_val, penalized)
    
    return penalized
