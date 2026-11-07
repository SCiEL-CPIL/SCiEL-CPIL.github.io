"""Guided local search penalty rule for the TSP – Adaptive penalty with hard exclusion for stuck edges."""
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill


def _compute_lower_bound(D):
    """MST-based lower bound on TSP tour length."""
    n = D.shape[0]
    if n < 2:
        return 0.0
    # Sum of two smallest edges incident to each node, divided by 2
    # This is a valid lower bound for TSP
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
    
    Uses a calibrated penalty based on the gap between current tour length
    and a lower bound, scaled by usage counts and annealed over rounds.
    Includes a 'hard exclusion' mechanism for edges that are frequently
    used in local optima, forcing the 2-opt pass to remove them.
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

    # --- Annealed penalty coefficient ---
    # Early rounds: larger penalty to explore. Later rounds: smaller to refine.
    # k starts at 1.8 and decays to 0.4 over 20 rounds
    k = 1.8 - 0.07 * min(round_num, 20)
    k = max(k, 0.4)

    # --- Build penalty matrix ---
    penalty = np.zeros((n, n), dtype=np.float64)

    # Base penalty per tour edge, scaled by gap
    base_pen = k * gap / max(len(tour), 1)
    
    # Determine a "hard exclusion" threshold.
    # If an edge has been used in more than 50% of the previous rounds, it's considered "stuck".
    # We set a very large penalty for these edges to force 2-opt to remove them.
    hard_threshold = 0.5 * round_num
    hard_penalty = 100.0 * gap  # Large enough to dominate 2-opt decisions

    for u, v in tour_edges:
        usage = float(U[u, v])
        
        if usage > hard_threshold:
            # Hard exclusion: make this edge very expensive
            ep = hard_penalty
        else:
            # Standard adaptive penalty
            # Use sqrt(usage) for sub-linear growth
            usage_factor = 1.0 + 0.3 * np.sqrt(usage)
            ep = base_pen * usage_factor
            
        penalty[u, v] += ep
        penalty[v, u] += ep

    # --- Global penalty for high-usage non-tour edges ---
    # Discourage re-forming past tours
    global_coeff = 0.08 * gap / max(n, 1)
    
    # Soft usage: sqrt(u) / (1 + sqrt(u)) for bounded saturation
    sqrt_U = np.sqrt(np.maximum(U, 0))
    soft_U = sqrt_U / (1.0 + sqrt_U)

    # Mask: not current tour, not diagonal
    mask = np.ones((n, n), dtype=bool)
    for u, v in tour_edges:
        mask[u, v] = False
        mask[v, u] = False
    np.fill_diagonal(mask, False)

    global_pen = np.where(mask & (soft_U > 0.1),
                          global_coeff * soft_U, 0.0)
    penalty += global_pen

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
