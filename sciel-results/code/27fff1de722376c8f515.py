"""Guided local search penalty rule for the TSP – geometrically-aware repulsion."""
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP.
    
    Uses a geometrically-aware repulsion penalty: edges are penalized in
    proportion to the product of their usage count and their relative rank
    among the shortest available edges. This specifically penalizes frequently
    used edges that are geometrically "easy" (short) to break, while leaving
    long, rarely used edges mostly unperturbed.
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

    # --- Infer round number ---
    max_used = int(U.max()) if U.size > 0 else 0
    round_num = max_used + 1  # 1-based

    # --- Annealed penalty coefficient ---
    # Early rounds: larger penalty to explore. Later rounds: smaller to refine.
    k = 1.5 - 0.05 * min(round_num, 20)
    k = max(k, 0.3)

    # --- Compute geometric rank factor for each edge ---
    # For each edge (i,j), compute its rank among all edges incident to i (or j)
    # A short edge has rank ~1, a long edge has rank ~n-1
    # We use: rank_factor = 1 / (1 + rank) so short edges get high factor
    # This makes the penalty proportional to usage * rank_factor
    
    # Efficient computation: for each node, rank its incident edges
    # rank_factor[i,j] = 1/(1 + rank_of_edge_ij_among_i's_edges)
    rank_factor = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        row = D[i, :].copy()
        row[i] = np.inf
        # Get ranks: rank 0 = shortest, rank n-2 = longest
        # Use argsort to get order, then assign ranks
        order = np.argsort(row)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n)
        ranks[i] = 0.0  # diagonal doesn't matter
        # rank_factor: 1/(1+rank) so shortest gets 1, longest gets ~1/n
        rf = 1.0 / (1.0 + ranks)
        rank_factor[i, :] = rf
        rank_factor[:, i] = rf  # symmetric
    
    # Symmetrize: take the average of both endpoints' rank factors
    rank_factor = (rank_factor + rank_factor.T) * 0.5

    # --- Soft usage: u/(1+u) for bounded saturation ---
    soft_U = U / (1.0 + U)

    # --- Build penalty: proportional to soft_usage * rank_factor ---
    # Only apply to off-diagonal edges
    np.fill_diagonal(rank_factor, 0.0)
    np.fill_diagonal(soft_U, 0.0)
    
    # Scale by gap to make penalty meaningful relative to tour length
    gap = tour_len / max(len(tour), 1)  # average edge length as scale
    
    # Penalty = k * gap * soft_usage * rank_factor
    # This penalizes edges that are both frequently used AND geometrically short
    penalty = k * gap * soft_U * rank_factor

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
