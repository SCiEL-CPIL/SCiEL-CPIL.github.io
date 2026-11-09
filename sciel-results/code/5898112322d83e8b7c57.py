"""Guided local search penalty rule for TSP with 100 points."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 325.0
_start_time = None


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.

    Strategy (combined):
    1. Rank-based quadratic penalty on edge usage (smooth convex barrier
       against the most-used edges, leaving low-usage edges nearly free).
    2. Strong penalty on current tour edges to escape the local optimum.
    3. Edge-reversal bias: for each tour edge, identify the best 2-opt
       replacement and add a small penalty to the original edge to nudge
       2-opt toward that specific move.
    4. Small degree-based topological bias to penalize edges incident to
       high-degree nodes in the used-edge graph.
    5. Adaptive calibration based on tour quality vs. theoretical lower bound.

    Returns a finite distance matrix of the same shape.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        return edge_distance.copy()

    n = edge_distance.shape[0]
    D = edge_distance.astype(np.float64).copy()
    tour = list(local_opt_tour)
    tour_len = len(tour)

    if tour_len < 4:
        return D

    # --- Reference scale ---
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_diag = D[off_diag_mask]
    scale = float(np.median(off_diag)) if off_diag.size > 0 else 1.0
    if scale < 1e-10:
        scale = 1.0

    # --- Tour edge mask ---
    tour_edge_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(tour_len):
        u = tour[i]
        v = tour[(i + 1) % tour_len]
        tour_edge_mask[u, v] = 1.0
        tour_edge_mask[v, u] = 1.0

    # --- Usage normalization ---
    usage = edge_n_used.astype(np.float64)
    max_usage = float(usage.max()) if usage.size > 0 else 0.0
    if max_usage < 1e-10:
        max_usage = 1.0
    usage_norm = usage / max_usage

    # --- Tour length and quality ratio ---
    tour_length = 0.0
    for i in range(tour_len):
        tour_length += D[tour[i], tour[(i + 1) % tour_len]]
    lower_bound = 0.7 * np.sqrt(n)
    quality_ratio = tour_length / lower_bound if lower_bound > 0 else 1.0

    # --- Mean usage of current tour edges ---
    tour_edge_usages = usage_norm[tour_edge_mask > 0.5]
    mean_tour_usage = float(tour_edge_usages.mean()) if tour_edge_usages.size > 0 else 0.0

    # --- Adaptive multiplier ---
    adaptive_mult = 1.0 + 0.5 * mean_tour_usage + 0.3 * max(0.0, quality_ratio - 1.0)

    # Base scale for penalties (fraction of typical edge distance)
    base_scale = scale * 0.12

    # ============================================================
    # Component 1: Rank-based quadratic penalty on usage
    # Sort edges by usage count; apply convex (quadratic) penalty
    # based on normalized rank. Steep barrier on most-used edges,
    # gentle on low-usage edges.
    # ============================================================
    # Flatten usage, get ranks (0 = least used, n*(n-1)-1 = most used)
    usage_flat = usage.ravel()
    # argsort gives indices that would sort; use argsort of argsort for rank
    order = np.argsort(usage_flat)
    rank = np.zeros_like(order)
    rank[order] = np.arange(usage_flat.size)
    rank = rank.reshape(n, n)

    # Normalize rank to [0, 1]
    max_rank = float(rank.max()) if rank.size > 0 else 1.0
    if max_rank < 1e-10:
        max_rank = 1.0
    rank_norm = rank / max_rank

    # Quadratic penalty: (rank_norm)^2 — convex, steep at top
    rank_penalty = base_scale * adaptive_mult * 0.8 * (rank_norm ** 2)
    # Zero out diagonal
    np.fill_diagonal(rank_penalty, 0.0)

    # ============================================================
    # Component 2: Strong penalty on current tour edges
    # ============================================================
    tour_penalty = base_scale * adaptive_mult * 1.2 * tour_edge_mask

    # ============================================================
    # Component 3: Edge-reversal bias
    # For each tour edge, find the best 2-opt replacement and add
    # a small penalty to the original edge to nudge 2-opt toward
    # that specific move.
    # ============================================================
    reversal_penalty = np.zeros((n, n), dtype=np.float64)

    check_time = time.time() - _start_time
    if check_time < _DEADLINE - 10.0:
        K = min(12, n - 1)
        neighbor_order = np.argsort(D, axis=1)

        for i in range(tour_len):
            u = tour[i]
            v = tour[(i + 1) % tour_len]

            pred_u = tour[(i - 1) % tour_len]
            succ_v = tour[(i + 2) % tour_len]

            best_delta = np.inf
            best_pair = None

            for c_idx in range(K):
                c = neighbor_order[u, c_idx]
                if c == v or c == pred_u:
                    continue
                for d_idx in range(K):
                    d = neighbor_order[v, d_idx]
                    if d == u or d == succ_v or d == c:
                        continue
                    if tour_edge_mask[c, d] > 0.5:
                        continue

                    delta1 = D[u, c] + D[v, d] - D[u, v] - D[c, d]
                    delta2 = D[u, d] + D[v, c] - D[u, v] - D[c, d]
                    delta = min(delta1, delta2)

                    if delta < best_delta:
                        best_delta = delta
                        best_pair = (c, d, delta1, delta2)

            if best_pair is not None and best_delta < 0:
                strength = min(-best_delta / scale, 1.0) * base_scale * 0.35
                reversal_penalty[u, v] += strength
                reversal_penalty[v, u] += strength

    # ============================================================
    # Component 4: Degree-based topological bias
    # ============================================================
    degree = usage.sum(axis=1)
    max_degree = float(degree.max()) if degree.size > 0 else 0.0
    degree_penalty = np.zeros((n, n), dtype=np.float64)
    if max_degree > 1e-10:
        norm_degree = degree / max_degree
        degree_penalty = (norm_degree[:, None] + norm_degree[None, :]) * 0.5
        degree_penalty *= base_scale * 0.25 * adaptive_mult
        np.fill_diagonal(degree_penalty, 0.0)

    # --- Combine all penalties ---
    total_penalty = rank_penalty + tour_penalty + reversal_penalty + degree_penalty

    # Enforce symmetry
    total_penalty = (total_penalty + total_penalty.T) / 2.0
    np.fill_diagonal(total_penalty, 0.0)

    # Apply
    D_out = D + total_penalty

    # Ensure finiteness and symmetry
    D_out = np.nan_to_num(D_out, nan=0.0, posinf=scale * 10, neginf=0.0)
    D_out = (D_out + D_out.T) / 2.0
    np.fill_diagonal(D_out, 0.0)

    return D_out
# EVOLVE-BLOCK-END
