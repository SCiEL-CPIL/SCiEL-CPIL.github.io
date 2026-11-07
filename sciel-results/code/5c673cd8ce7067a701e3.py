import numpy as np
import time

_start_time = time.monotonic()
_DEADLINE = 325.0  # safety margin before 330s kill


def _elapsed():
    return time.monotonic() - _start_time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided Local Search penalty for TSP.
    
    Combines:
    - Focused penalty on current tour edges (stronger weight)
    - Usage-aware penalty on all previously used edges (weaker weight)
    - Smooth log-saturating penalty scaled by median distance
    """
    # Deadline guard
    if _elapsed() > _DEADLINE:
        return np.asarray(edge_distance, dtype=float).copy()
    
    n = edge_distance.shape[0]
    
    # Compute scale factor: median of non-diagonal distances
    mask = ~np.eye(n, dtype=bool)
    nonzero_mask = (edge_distance > 1e-10) & mask
    if np.any(nonzero_mask):
        dist_scale = np.median(edge_distance[nonzero_mask])
    else:
        dist_scale = 1.0
    
    if dist_scale <= 0 or not np.isfinite(dist_scale):
        dist_scale = 1.0
    
    usage = edge_n_used.astype(np.float64)
    
    # Build penalty matrix: combine focused tour-edge penalty + usage penalty
    # Focused penalty: stronger on edges in the current local-opt tour
    tour_penalty = np.zeros((n, n), dtype=np.float64)
    tour = np.asarray(local_opt_tour, dtype=np.intp)
    for i in range(len(tour) - 1):
        a, b = int(tour[i]), int(tour[i + 1])
        tour_penalty[a, b] = 1.0
        tour_penalty[b, a] = 1.0
    # Close the tour
    a, b = int(tour[-1]), int(tour[0])
    tour_penalty[a, b] = 1.0
    tour_penalty[b, a] = 1.0
    
    # Smooth saturating penalty from usage: log1p(usage) / (1 + log1p(usage))
    # This gives values in [0, 1) with diminishing returns
    usage_penalty_raw = np.log1p(usage)
    usage_penalty = usage_penalty_raw / (1.0 + usage_penalty_raw)  # saturates toward 1
    
    # Combine: alpha_tour * tour_penalty + alpha_usage * usage_penalty
    # Tune: tour edges get stronger penalty to push 2-opt away from them
    alpha_tour = 0.35
    alpha_usage = 0.15
    
    combined = alpha_tour * tour_penalty + alpha_usage * usage_penalty
    
    # Scale by dist_scale so penalty magnitude is comparable to edge lengths
    # Use multiplicative form: result = edge_distance * (1 + scale * combined)
    # The scale factor ensures penalty is a meaningful fraction of typical distances
    scale_factor = dist_scale * 0.5
    
    result = edge_distance * (1.0 + scale_factor * combined)
    
    # Ensure finiteness and non-negativity
    result = np.where(np.isfinite(result), result, edge_distance)
    result = np.maximum(result, 1e-10)
    
    return result
