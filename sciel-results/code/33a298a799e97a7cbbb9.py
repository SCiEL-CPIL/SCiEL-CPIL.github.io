"""Guided local search penalty for TSP: Refined local scale (min2) + K-NN structural bias + sigmoid disruption."""
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

    # Refined Local Scale: 2nd smallest incident distance (from Inspiration 2)
    # This is a tighter estimate of "expected short edge" than median.
    diag_mask = np.eye(n, dtype=bool)
    d_work = d.copy()
    d_work[diag_mask] = np.inf
    
    # Find 2 smallest values per row
    # argsort is O(n^2 log n), acceptable for n=100
    order = np.argsort(d_work, axis=1)
    min1 = d_work[np.arange(n), order[:, 0]]
    min2 = d_work[np.arange(n), order[:, 1]]
    
    # Use min2 as the local scale. If min2 is 0 (duplicate points?), fall back to min1 or 1.0
    local_scale = min2
    local_scale = np.where(local_scale > 1e-12, local_scale, min1)
    local_scale = np.where(local_scale > 1e-12, local_scale, 1.0)

    # Over-length ratio for each tour edge
    # Denominator is sum of local scales of endpoints
    denom = local_scale[i] + local_scale[j]
    ratio = tour_len / denom

    # ---- Structural K-NN candidate set ----
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

    # Usage-based penalty: mild, log-scaled
    usage = np.asarray(edge_n_used, dtype=float)
    usage = np.maximum(usage, 0.0)
    u_max = usage.max()
    if u_max > 0:
        usage_scaled = np.log1p(usage) / np.log1p(u_max)
        penalty += 0.15 * usage_scaled  # Slightly increased from 0.10 to 0.15

    # ---- Combined disruption on tour edges ----
    # Smooth sigmoid weight on length ratio.
    # With min2 scale, ratios will be larger on average, so we might need to adjust center/steepness.
    # Previous center 1.0, steepness 5.0.
    # If ratios are now larger (e.g., typical short edge ratio ~2-3 instead of ~1), 
    # we should shift the center to avoid saturating the sigmoid at 1.0 for all edges.
    # Let's estimate typical ratio: min2 is ~ 0.1-0.2 for unit square? 
    # Actually, for 100 pts in unit square, min dist ~ 0.02, min2 ~ 0.03. 
    # Avg edge ~ 0.1. Ratio ~ 0.1 / (0.03+0.03) = 1.6.
    # So center 1.5 might be better? Or keep 1.0 and let it push harder?
    # Let's try center 1.2, steepness 4.0 to be slightly more selective on very long edges.
    center = 1.2
    steepness = 4.0
    weight = 1.0 / (1.0 + np.exp(-steepness * (ratio - center)))

    # Structural boost: edges that are NOT plausible candidates deserve extra disruption.
    tour_is_candidate = is_candidate[i, j]
    # Non-candidate edges: amplify their weight.
    struct_boost = np.where(tour_is_candidate, 0.0, 0.5)
    combined_weight = np.clip(weight + struct_boost, 0.0, 1.5)

    # Magnitude scales with edge length and excess over local scale.
    # excess_ratio = max(ratio - center, 0)
    excess_ratio = np.maximum(ratio - center, 0.0)
    # Base magnitude is proportional to the edge length itself to ensure penalty is in distance units.
    mag = 0.30 * tour_len + 0.30 * excess_ratio * mean_tour_edge

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
