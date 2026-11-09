"""Guided local search penalty rule for TSP: targeted tour-edge penalty plus
adaptive global usage penalty with count damping."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 320.0  # Safety margin before the 330s kill

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    GLS perturbation: add a targeted penalty to the current local-optimum tour's
    edges (forcing 2-opt to escape the current basin) plus a global penalty that
    grows with accumulated edge usage (discouraging repeatedly-visited edges).
    Both penalties are scaled by the instance's average edge length so the
    landscape perturbation is scale-invariant.
    """
    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()

    n = len(local_opt_tour)
    if n == 0:
        return edge_distance.copy()

    # Work on a copy so the input (true distances) is never mutated.
    penalized_dist = edge_distance.astype(float, copy=True)

    tour = np.asarray(local_opt_tour)
    u = tour
    v = np.roll(tour, -1)

    # Instance-geometry statistics for adaptive scaling.
    current_edge_lengths = edge_distance[u, v]
    tour_len = float(np.sum(current_edge_lengths))
    avg_edge_len = float(np.mean(current_edge_lengths))

    # Fallbacks for degenerate / non-finite geometry.
    if not np.isfinite(tour_len) or tour_len == 0:
        tour_len = float(np.sum(edge_distance)) / n
        avg_edge_len = tour_len / n
    if not np.isfinite(avg_edge_len) or avg_edge_len == 0:
        avg_edge_len = float(np.mean(edge_distance))
    if not np.isfinite(avg_edge_len) or avg_edge_len == 0:
        avg_edge_len = 0.1

    # --- Global usage penalty (structural idea from the inspirations) ---
    # Dampen high counts so early-round usage is not over-weighted and the
    # perturbation stays bounded as edge_n_used accumulates over 20 rounds.
    # sqrt(x+1)-1 == x / (sqrt(x+1)+1): grows like sqrt for large x, ~x for small.
    decayed_usage = np.sqrt(edge_n_used + 1.0) - 1.0
    alpha = 0.07
    penalized_dist += (alpha * avg_edge_len) * decayed_usage

    # --- Targeted penalty on current tour edges (current program's core) ---
    # Make the current local optimum noticeably suboptimal under the penalized
    # distances so the subsequent 2-opt must modify the tour to escape.
    margin = 0.08
    target_penalty_per_edge = (tour_len * margin) / n

    # Edges used more often in history get a slightly stronger push, so the
    # search prioritizes escaping the most "stuck" parts of the tour.
    edge_usage_in_tour = edge_n_used[u, v]
    max_usage = float(np.max(edge_n_used)) if np.max(edge_n_used) > 0 else 1.0
    usage_factor = 1.0 + 0.3 * (edge_usage_in_tour / max(1.0, max_usage))

    penalized_dist[u, v] += target_penalty_per_edge * usage_factor
    penalized_dist[v, u] += target_penalty_per_edge * usage_factor

    # Keep the matrix a valid symmetric metric for 2-opt.
    np.fill_diagonal(penalized_dist, 0.0)
    penalized_dist = (penalized_dist + penalized_dist.T) / 2.0

    # Guarantee a finite, non-negative output; anything else invalidates the run.
    if not np.all(np.isfinite(penalized_dist)):
        penalized_dist = np.nan_to_num(penalized_dist, nan=0.0,
                                       posinf=float(np.max(edge_distance)),
                                       neginf=0.0)
    penalized_dist = np.maximum(penalized_dist, 0.0)
    np.fill_diagonal(penalized_dist, 0.0)

    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()

    return penalized_dist
# EVOLVE-BLOCK-END
