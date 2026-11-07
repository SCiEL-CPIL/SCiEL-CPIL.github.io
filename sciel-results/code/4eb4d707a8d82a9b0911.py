"""Guided local search penalty rule for the TSP – dynamic edge-removal with soft threshold."""
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 315.0  # Safety margin before 330s kill


def _compute_lower_bound(D):
    """MST-based lower bound on TSP tour length."""
    n = D.shape[0]
    if n < 2:
        return 0.0
    min1 = np.full(n, np.inf)
    min2 = np.full(n, np.inf)
    for i in range(n):
        row = D[i, :].copy()
        row[i] = np.inf
        if n < 3:
            min1[i] = row.min()
            min2[i] = 0.0
        else:
            idx = np.argpartition(row, 2)[:2]
            min1[i] = row[idx].min()
            min2[i] = row[idx].max()
    return 0.5 * np.sum(min1 + min2)


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP.
    
    Uses dynamic edge-removal: edges with usage exceeding a round-dependent
    threshold get a large penalty, forcing reconstruction from fresh edges.
    Combined with smooth soft-min penalty on tour edges for selective perturbation.
    """
    # Time guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return edge_distance.copy()

    n = edge_distance.shape[0]
    tour = local_opt_tour

    if n < 2 or len(tour) < 2:
        return edge_distance.copy()

    D = np.asarray(edge_distance, dtype=np.float64)
    U = np.asarray(edge_n_used, dtype=np.float64)

    # --- Compute current tour length ---
    tour_len = 0.0
    tour_edges = []
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        tour_edges.append((u, v))
        tour_len += D[u, v]

    # --- Compute lower bound ---
    lb = _compute_lower_bound(D)
    gap = tour_len - lb
    if gap < 1e-9:
        gap = 1.0

    # --- Infer round number ---
    max_used = int(U.max()) if U.size > 0 else 0
    round_num = max_used + 1  # 1-based
    total_rounds = 20

    # --- Dynamic edge-removal threshold ---
    # Threshold increases over rounds: early rounds allow more edges, later rounds force fresh edges
    # threshold = 2 * round_index / total_rounds, where round_index is 0-based
    round_index = max_used  # 0-based round index
    threshold = 2.0 * round_index / total_rounds
    
    # --- Build penalty matrix ---
    penalty = np.zeros((n, n), dtype=np.float64)
    
    # Mean edge distance for scaling
    mean_edge = tour_len / max(len(tour), 1)
    
    # --- Smooth soft-min penalty on tour edges ---
    # Use a smooth surrogate: instead of hard removal, apply a smooth penalty
    # that grows rapidly when usage exceeds threshold
    # soft_penalty = mean_edge * (1 + alpha * exp(beta * (usage - threshold)))
    alpha = 0.5
    beta = 3.0
    
    for u, v in tour_edges:
        usage = float(U[u, v])
        # Smooth exponential penalty that kicks in when usage > threshold
        excess = max(usage - threshold, 0.0)
        soft_pen = mean_edge * alpha * np.exp(beta * excess)
        # Also add a base penalty for all tour edges, scaled by usage
        base_pen = mean_edge * 0.3 * (1.0 + 0.2 * np.sqrt(usage))
        ep = base_pen + soft_pen
        penalty[u, v] += ep
        penalty[v, u] += ep

    # --- Global penalty for high-usage non-tour edges ---
    # Discourage re-forming past tours
    global_coeff = 0.1 * mean_edge
    
    # Soft usage: smooth saturation
    sqrt_U = np.sqrt(np.maximum(U, 0))
    soft_U = sqrt_U / (1.0 + sqrt_U)
    
    # Mask: not current tour, not diagonal
    mask = np.ones((n, n), dtype=bool)
    for u, v in tour_edges:
        mask[u, v] = False
        mask[v, u] = False
    np.fill_diagonal(mask, False)
    
    # Apply global penalty to non-tour edges with significant usage
    global_pen = np.where(mask & (soft_U > 0.15),
                          global_coeff * soft_U, 0.0)
    penalty += global_pen

    # --- Dynamic edge-removal: large penalty for edges exceeding threshold ---
    # For all edges (not just tour edges), if usage > threshold, add large penalty
    # This forces the 2-opt to avoid these edges
    large_pen_coeff = 5.0 * mean_edge
    usage_mask = (U > threshold) & (U > 0)
    np.fill_diagonal(usage_mask, False)
    
    # Only apply to edges that are not in the current tour (they already get penalty above)
    tour_mask = np.zeros((n, n), dtype=bool)
    for u, v in tour_edges:
        tour_mask[u, v] = True
        tour_mask[v, u] = True
    
    removal_mask = usage_mask & ~tour_mask
    penalty[removal_mask] += large_pen_coeff

    # --- Combine ---
    result = D + penalty

    # Ensure symmetry
    result = (result + result.T) * 0.5
    # Ensure finite
    result = np.nan_to_num(result, nan=0.0,
                           posinf=np.finfo(np.float64).max / 10,
                           neginf=0.0)
    # Zero diagonal
    np.fill_diagonal(result, 0.0)

    return result
