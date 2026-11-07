import numpy as np
import time

_start_time = time.monotonic()
_DEADLINE = 325.0


def _elapsed():
    return time.monotonic() - _start_time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided Local Search penalty for TSP with crowding-aware repulsion.
    
    Combines:
    - Usage-aware penalty on current tour edges with soft saturation
    - Crowding cost: penalize edges in dense regions to encourage spatial uniformity
    - Small uniform penalty for exploration
    - Smooth, moderate penalty magnitudes scaled by median distance
    """
    # Deadline guard
    if _elapsed() > _DEADLINE:
        return np.asarray(edge_distance, dtype=float).copy()
    
    D = np.asarray(edge_distance, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1] or D.shape[0] < 3:
        return D.copy()
    
    n = D.shape[0]
    
    # Guard against non-finite input
    if not np.all(np.isfinite(D)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        return base
    
    used = np.asarray(edge_n_used, dtype=float)
    if used.shape != D.shape:
        used = np.zeros_like(D)
    if not np.all(np.isfinite(used)):
        used = np.nan_to_num(used, nan=0.0, posinf=0.0, neginf=0.0)
    used = np.maximum(used, 0.0)
    
    # Reference scale: median of non-diagonal distances
    mask = ~np.eye(n, dtype=bool)
    nonzero_mask = (D > 1e-10) & mask
    if np.any(nonzero_mask):
        ref = float(np.median(D[nonzero_mask]))
    else:
        ref = 1.0
    
    if not np.isfinite(ref) or ref <= 0.0:
        ref = 1.0
    
    # Parse tour edges
    tour = np.asarray(local_opt_tour, dtype=int)
    tour_edges = []
    if tour.ndim == 1 and len(tour) >= 2:
        for i in range(len(tour) - 1):
            a, b = int(tour[i]), int(tour[i + 1])
            if 0 <= a < n and 0 <= b < n:
                tour_edges.append((a, b))
        a, b = int(tour[-1]), int(tour[0])
        if 0 <= a < n and 0 <= b < n:
            tour_edges.append((a, b))
    
    # Start with zero penalty matrix
    penalty = np.zeros_like(D)
    
    # Compute local point density around each node using distance matrix
    # For each node, compute the mean distance to its k-nearest neighbors
    # Lower mean distance = denser region
    k = max(3, n // 10)
    if k > n - 1:
        k = n - 1
    
    # For each node, find mean distance to k nearest neighbors
    density_scores = np.zeros(n)
    for i in range(n):
        dists = D[i].copy()
        dists[i] = np.inf  # exclude self
        # Get k smallest distances
        if k < n:
            kth_dist = np.partition(dists, k)[k]
            # Mean of distances <= kth_dist (excluding self)
            near_mask = dists <= kth_dist
            if np.any(near_mask):
                density_scores[i] = np.mean(dists[near_mask])
            else:
                density_scores[i] = np.mean(dists)
        else:
            density_scores[i] = np.mean(dists)
    
    # Normalize density scores: lower score = denser region
    # We want to penalize edges in dense regions more
    if np.any(density_scores > 0):
        min_density = np.min(density_scores)
        max_density = np.max(density_scores)
        if max_density > min_density:
            # Normalize to [0, 1], where 0 = densest, 1 = sparsest
            norm_density = (density_scores - min_density) / (max_density - min_density)
        else:
            norm_density = np.zeros(n)
    else:
        norm_density = np.zeros(n)
    
    # Apply usage-aware penalty to tour edges
    tour_penalty_base = 0.25 * ref
    if tour_edges:
        for (a, b) in tour_edges:
            u = 0.5 * (used[a, b] + used[b, a])
            # Soft saturation: grows linearly at first, then saturates
            scaled_usage = u / (1.0 + u)
            pen = tour_penalty_base * scaled_usage
            penalty[a, b] += pen
            penalty[b, a] += pen
    
    # Apply crowding penalty: penalize edges in dense regions
    # Higher density (lower norm_density) -> higher penalty
    crowd_base = 0.08 * ref
    for i in range(n):
        for j in range(i + 1, n):
            # Edge crowding: average density of endpoints
            edge_crowd = 0.5 * (norm_density[i] + norm_density[j])
            # Inverse: dense regions get higher penalty
            pen_crowd = crowd_base * (1.0 - edge_crowd)
            penalty[i, j] += pen_crowd
            penalty[j, i] += pen_crowd
    
    # Add a small uniform penalty to all non-diagonal elements
    small_uniform = 0.005 * ref
    penalty += small_uniform
    np.fill_diagonal(penalty, 0.0)
    
    # Hard cap to prevent over-penalization
    max_penalty = 0.5 * ref
    penalty = np.minimum(penalty, max_penalty)
    penalty = np.maximum(penalty, 0.0)
    
    # Add penalty to the distance matrix
    out = D + penalty
    
    # Enforce exact symmetry
    out = 0.5 * (out + out.T)
    
    # Ensure diagonal is zero
    np.fill_diagonal(out, 0.0)
    
    # Final finiteness check
    if not np.all(np.isfinite(out)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        out = 0.5 * (base + base.T)
        np.fill_diagonal(out, 0.0)
    
    return out
