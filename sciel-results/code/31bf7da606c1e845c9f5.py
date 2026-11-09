"""Guided local search penalty for TSP: geometric curvature + usage disruption."""
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 305.0  # Safety margin before 330s kill


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

    # Compute local scale: for each node, the median of its incident distances
    diag_mask = np.eye(n, dtype=bool)
    d_work = d.copy()
    d_work[diag_mask] = np.inf

    # For each node, compute the median of incident distances (excluding self)
    local_scale = np.median(d_work, axis=1)
    # Guard against zero
    local_scale = np.where(local_scale > 1e-12, local_scale, 1.0)

    # Over-length ratio for each tour edge
    ratio = tour_len / (local_scale[i] + local_scale[j])

    # Build penalty matrix
    penalty = np.zeros((n, n), dtype=float)

    # Usage-based penalty: penalize edges that have been used many times
    usage = np.asarray(edge_n_used, dtype=float)
    usage = np.maximum(usage, 0.0)
    u_max = usage.max()
    if u_max > 0:
        # Smooth saturation: usage/(1+usage) gives bounded, smooth penalty
        # Scale by a factor to make it meaningful relative to distances
        usage_penalty = 0.08 * usage / (1.0 + usage)
        penalty += usage_penalty
    else:
        penalty += np.zeros_like(usage)

    # Geometric curvature penalty: for each node in the tour, compute the angle
    # between its two incident edges. Penalize deviation from 180 degrees.
    # This encourages 2-opt to find more "straight" paths through nodes.
    
    # For node k in the tour, its neighbors are tour[(k-1)%n] and tour[(k+1)%n]
    # The angle is between vectors (tour[k] - tour[k-1]) and (tour[k+1] - tour[k])
    # We don't have coordinates, but we can use the distance matrix to compute
    # a proxy: the "defect" is how much the sum of the two incident edge lengths
    # exceeds the direct distance between the two neighbors.
    
    # For node k: neighbors are prev = tour[(k-1)%n] and next = tour[(k+1)%n]
    # Edges: (prev, k) and (k, next)
    # Defect = d[prev,k] + d[k,next] - d[prev,next]
    # This is >= 0 by triangle inequality, and is large when the path through k
    # is much longer than the direct connection (i.e., k is a "detour")
    
    prev_nodes = np.roll(tour, 1)
    next_nodes = np.roll(tour, -1)
    
    # d[prev, k] is the edge from prev to k
    # d[k, next] is the edge from k to next
    # d[prev, next] is the direct distance between neighbors
    
    edge1 = d[prev_nodes, tour]  # d[prev, k]
    edge2 = d[tour, next_nodes]  # d[k, next]
    direct = d[prev_nodes, next_nodes]  # d[prev, next]
    
    # Defect: how much the two-edge path exceeds the direct path
    defect = edge1 + edge2 - direct
    defect = np.maximum(defect, 0.0)
    
    # Normalize defect by the local scale to make it comparable
    node_scale = (local_scale[prev_nodes] + local_scale[tour] + local_scale[next_nodes]) / 3.0
    node_scale = np.where(node_scale > 1e-12, node_scale, 1.0)
    normalized_defect = defect / node_scale
    
    # Penalize nodes with high curvature (large defect)
    # Add penalty to both incident edges of the high-curvature node
    curvature_penalty = np.zeros((n, n), dtype=float)
    for k in range(n):
        node = tour[k]
        prev = prev_nodes[k]
        next_ = next_nodes[k]
        nd = normalized_defect[k]
        if nd > 0.1:  # Only penalize significant curvature
            # Penalty magnitude scales with the defect
            p = 0.15 * nd * (edge1[k] + edge2[k])
            curvature_penalty[node, prev] += p
            curvature_penalty[prev, node] += p
            curvature_penalty[node, next_] += p
            curvature_penalty[next_, node] += p
    
    penalty += curvature_penalty

    # Smooth sigmoid-weighted disruption on over-long edges
    center = 1.0
    steepness = 5.0
    weight = 1.0 / (1.0 + np.exp(-steepness * (ratio - center)))

    # Penalty magnitude scales with how much the edge exceeds local scale
    excess_ratio = np.maximum(ratio - center, 0.0)
    mag = 0.25 * tour_len + 0.15 * mean_tour_edge + 0.35 * excess_ratio * mean_tour_edge

    # Apply smooth penalty to all tour edges, weighted by sigmoid
    for idx in range(n):
        a, b = i[idx], j[idx]
        p = weight[idx] * mag[idx]
        penalty[a, b] += p
        penalty[b, a] += p

    # Anti-clustering: penalize edges that are very short relative to local scale
    # These are "obvious" edges that 2-opt keeps re-adding; nudge exploration
    short_ratio = 1.0 / np.maximum(ratio, 1e-6)
    short_center = 1.5
    short_steepness = 4.0
    short_weight = 1.0 / (1.0 + np.exp(short_steepness * (short_ratio - short_center)))
    for idx in range(n):
        if short_weight[idx] > 0.15:
            a, b = i[idx], j[idx]
            p = 0.05 * short_weight[idx] * tour_len[idx]
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
