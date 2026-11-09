"""Guided local search penalty rule for TSP with topological disruption."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 320.0

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Topological disruption GLS: identify the k shortest edges in the current
    tour and set their distances to a large value, forcing 2-opt to restructure
    the tour around alternative connections. Also applies a mild penalty to
    remaining tour edges scaled by usage count.
    """
    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()

    n = len(local_opt_tour)
    if n == 0:
        return edge_distance.copy()

    penalized_dist = edge_distance.copy()

    tour = np.asarray(local_opt_tour)
    u = tour
    v = np.roll(tour, -1)

    # Get true distances for these tour edges
    current_edge_lengths = edge_distance[u, v]

    # Fallback for degenerate cases
    avg_edge_len = np.mean(current_edge_lengths)
    if not np.isfinite(avg_edge_len) or avg_edge_len == 0:
        avg_edge_len = np.mean(edge_distance)
        if not np.isfinite(avg_edge_len) or avg_edge_len == 0:
            avg_edge_len = 0.1

    max_dist = np.max(edge_distance)
    if not np.isfinite(max_dist) or max_dist == 0:
        max_dist = 1.0

    # --- Topological disruption: remove k shortest tour edges ---
    # Choose k adaptively: more disruption when edges are heavily used
    # k is between 2 and max(3, n//10)
    k_min = 2
    k_max = max(3, n // 10)

    # Use edge_n_used to adapt k: higher usage -> more disruption
    tour_usage = edge_n_used[u, v]
    avg_usage = np.mean(tour_usage) if tour_usage.size > 0 else 1.0

    # Scale k based on average usage (capped)
    usage_factor = min(avg_usage / 5.0, 1.0) if avg_usage > 0 else 0.0
    k = k_min + int(round(usage_factor * (k_max - k_min)))
    k = max(k_min, min(k, n))

    # Find the k shortest edges in the current tour
    # Sort tour edges by length
    sorted_idx = np.argsort(current_edge_lengths)
    k_idx = sorted_idx[:k]

    # Set these edges to a large distance (effectively removing them)
    # Use a value well above the max so 2-opt avoids them
    large_val = max_dist * 10.0 + avg_edge_len
    for idx in k_idx:
        i, j = u[idx], v[idx]
        penalized_dist[i, j] = large_val
        penalized_dist[j, i] = large_val

    # --- Mild penalty on remaining tour edges, scaled by usage ---
    # For edges NOT in the disrupted set, add a smaller penalty
    remaining_mask = np.ones(n, dtype=bool)
    remaining_mask[k_idx] = False

    if np.any(remaining_mask):
        rem_u = u[remaining_mask]
        rem_v = v[remaining_mask]
        rem_usage = tour_usage[remaining_mask]
        rem_lengths = current_edge_lengths[remaining_mask]

        # Penalty proportional to usage and relative length
        # Higher usage -> higher penalty (discourages repeated edges)
        # Use log-smoothed scaling for gradual increase
        usage_scaled = np.log1p(rem_usage) / np.log1p(max(1.0, np.max(rem_usage)))
        # Base penalty: fraction of avg edge length, scaled by usage
        base_penalty = 0.3 * avg_edge_len
        penalties = base_penalty * (0.5 + 0.5 * usage_scaled)

        # Also scale by relative edge length (shorter edges get slightly less penalty
        # since they're more likely to be part of good solutions)
        rel_len = rem_lengths / (avg_edge_len + 1e-10)
        rel_factor = 0.8 + 0.2 * np.clip(rel_len, 0.5, 1.5)
        penalties *= rel_factor

        penalized_dist[rem_u, rem_v] += penalties
        penalized_dist[rem_v, rem_u] += penalties

    # Ensure the diagonal is 0 to avoid self-loops
    np.fill_diagonal(penalized_dist, 0.0)

    # Ensure symmetry
    penalized_dist = (penalized_dist + penalized_dist.T) / 2.0

    # Check for finiteness
    if not np.all(np.isfinite(penalized_dist)):
        penalized_dist = np.nan_to_num(penalized_dist, nan=0.0, posinf=max_dist * 10, neginf=0.0)

    return penalized_dist
# EVOLVE-BLOCK-END
