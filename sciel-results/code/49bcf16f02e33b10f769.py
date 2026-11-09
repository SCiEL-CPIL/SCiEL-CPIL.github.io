"""Guided local search penalty rule for TSP with sparse tour-edge penalties."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Sparse GLS penalty: add a constant penalty to edges in the current tour,
    scaled by average edge length. This forces the local search to escape
    the current basin by modifying the cost landscape only where stuck.
    """
    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()

    n = len(local_opt_tour)
    if n == 0:
        return edge_distance.copy()

    # Create a copy of the true distance matrix to avoid mutating input
    penalized_dist = edge_distance.copy()

    # Extract edges of the current local optimum tour
    tour = np.asarray(local_opt_tour)
    u = tour
    v = np.roll(tour, -1)

    # Get true distances for these tour edges
    current_edge_lengths = edge_distance[u, v]
    avg_edge_len = np.mean(current_edge_lengths)

    # Fallback for degenerate cases
    if not np.isfinite(avg_edge_len) or avg_edge_len == 0:
        avg_edge_len = np.mean(edge_distance)
        if not np.isfinite(avg_edge_len) or avg_edge_len == 0:
            avg_edge_len = 0.1

    # Adaptive penalty: scale each edge's penalty by its usage count and
    # relative to its own length. Edges used more often get stronger penalties.
    # This makes the perturbation proportional to how "stuck" each edge is.
    usage_counts = edge_n_used[u, v].astype(float)
    
    # Normalize usage to get a multiplier in [1, max_usage]
    max_usage = np.max(usage_counts) if np.any(usage_counts > 0) else 1.0
    if max_usage > 0:
        usage_multiplier = 1.0 + usage_counts / max_usage
    else:
        usage_multiplier = np.ones(n)
    
    # Per-edge penalty: base fraction of that edge's own length, scaled by usage
    # 0.3 base coefficient, up to 2x for most-used edges
    per_edge_penalty = 0.3 * current_edge_lengths * usage_multiplier
    
    penalized_dist[u, v] += per_edge_penalty
    penalized_dist[v, u] += per_edge_penalty

    # Ensure the diagonal is 0 to avoid self-loops
    np.fill_diagonal(penalized_dist, 0.0)

    # Ensure symmetry (already symmetric from construction, but be safe)
    penalized_dist = (penalized_dist + penalized_dist.T) / 2.0
    
    # Clip any values that might have become negative due to numerical issues
    penalized_dist = np.maximum(penalized_dist, 0.0)

    # Check for finiteness
    if not np.all(np.isfinite(penalized_dist)):
        penalized_dist = np.nan_to_num(penalized_dist, nan=0.0, posinf=np.max(edge_distance), neginf=0.0)

    return penalized_dist
# EVOLVE-BLOCK-END
