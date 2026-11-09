"""Guided local search penalty for TSP: K-NN structural bias + sigmoid length-ratio disruption."""
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 315.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Deadline guard
    if time.perf_counter() > _DEADLINE:
        return np.asarray(edge_distance, dtype=float).copy()

    n = edge_distance.shape[0]
    if n <= 2:
        return np.asarray(edge_distance, dtype=float).copy()

    # True distances
    d = np.asarray(edge_distance, dtype=float)

    # Build the edge list of the current tour (closed cycle)
    tour = np.asarray(local_opt_tour, dtype=int)
    if tour.shape[0] != n:
        tour = np.arange(n)

    # Edge endpoints for the closed tour
    i = tour[:-1]
    j = tour[1:]
    i = np.append(i, tour[0])
    j = np.append(j, tour[-1])

    # Lengths of each tour edge (true)
    tour_len = d[i, j]
    mean_tour_edge = np.mean(tour_len)

    # Local scale: per-node median of incident distances (robust "expected short edge")
    diag_mask = np.eye(n, dtype=bool)
    d_work = d.copy()
    d_work[diag_mask] = np.inf
    local_scale = np.median(d_work, axis=1)
    local_scale = np.where(local_scale > 1e-12, local_scale, 1.0)

    # Over-length ratio for each tour edge
    ratio = tour_len / (local_scale[i] + local_scale[j])

    # ---- Structural K-NN candidate set (inspiration 2) ----
    # An edge is "plausible" if either endpoint ranks the other among its K nearest.
    K = 6
    d_sorted_idx = np.argsort(d, axis=1)
    nn_mask = np.zeros((n, n), dtype=bool)
    for u in range(n):
        count = 0
        for v in d_sorted_idx[u, :K + 1]:
            if v == u:
                continue
            nn_mask[u, v] = True
            count += 1
            if count >= K:
                break
    is_candidate = nn_mask | nn_mask.T

    # Build penalty matrix
    penalty = np.zeros((n, n), dtype=float)

    # Usage-based penalty: mild, log-scaled, to prevent cycling
    usage = np.asarray(edge_n_used, dtype=float)
    usage = np.maximum(usage, 0.0)
    u_max = usage.max()
    if u_max > 0:
        usage_scaled = np.log1p(usage) / np.log1p(u_max)
        penalty += 0.10 * usage_scaled

    # ---- Combined disruption on tour edges (current + sigmoid refinement) ----
    # Smooth sigmoid weight on length ratio: near 0 for short edges, ~1 for long ones.
    center = 1.0
    steepness = 5.0
    weight = 1.0 / (1.0 + np.exp(-steepness * (ratio - center)))

    # Structural boost: edges that are NOT plausible candidates deserve extra disruption.
    # Vectorized: candidate tour edges get a reduced weight, non-candidates full/boosted.
    tour_is_candidate = is_candidate[i, j]
    # Non-candidate edges: amplify their weight (up to +0.5) so 2-opt breaks them.
    struct_boost = np.where(tour_is_candidate, 0.0, 0.5)
    combined_weight = np.clip(weight + struct_boost, 0.0, 1.5)

    # Magnitude scales with edge length, mean edge, and how far it exceeds local scale.
    excess_ratio = np.maximum(ratio - center, 0.0)
    mag = 0.30 * tour_len + 0.20 * mean_tour_edge + 0.40 * excess_ratio * mean_tour_edge

    for idx in range(n):
        a, b = i[idx], j[idx]
        p = combined_weight[idx] * mag[idx]
        penalty[a, b] += p
        penalty[b, a] += p

    # Symmetrize
    penalty = 0.5 * (penalty + penalty.T)

    # Result: true distance + penalty
    result = d + penalty

    # Ensure finiteness
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=0.0)

    return result
