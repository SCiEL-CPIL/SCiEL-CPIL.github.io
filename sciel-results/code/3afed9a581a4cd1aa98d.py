"""Guided local search penalty for TSP; directed edge-reversal penalty with multi-strategy search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 300.0
_start_time = None


def _two_opt(D, tour, max_passes=10):
    """2-opt on distance matrix D starting from tour. Returns improved tour."""
    n = len(tour)
    if n < 4:
        return list(tour)
    tour = list(tour)
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 1):
            a, b = tour[i - 1], tour[i]
            for j in range(i + 1, n):
                c, d = tour[j], tour[(j + 1) % n]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < -1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return tour


def _closed_len(D, tour):
    """Compute the length of the closed tour using distance matrix D."""
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += D[tour[i], tour[(i + 1) % n]]
    return total


def _compute_edge_improvements(edge_distance, tour):
    """For each tour edge, compute the best 2-opt improvement if that edge were removed.
    Returns a vector of improvement magnitudes (positive = edge is costly)."""
    n = len(tour)
    improvements = np.zeros(n)
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        # Best 2-opt move that removes edge (a,b): find j such that
        # removing (a,b) and (c,d) and adding (a,c) and (b,d) improves.
        # The edge (a,b) is at position i, so the other edge must be at position j != i.
        # For 2-opt: reverse segment between i and j.
        # delta = D[a, c] + D[b, d] - D[a, b] - D[c, d] where c=tour[j], d=tour[(j+1)%n]
        # We want the most negative delta (best improvement) among all j.
        best_delta = 0.0
        for j in range(n):
            if j == i or j == (i - 1) % n:
                continue
            c = tour[j]
            d = tour[(j + 1) % n]
            delta = edge_distance[a, c] + edge_distance[b, d] - edge_distance[a, b] - edge_distance[c, d]
            if delta < best_delta:
                best_delta = delta
        improvements[i] = -best_delta  # positive means removing this edge helps
    return improvements


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n = edge_distance.shape[0]
    tour = list(local_opt_tour)

    # Check if we're close to the deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5:
        max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
        return edge_distance * (1.0 + 0.1 * edge_n_used / (1.0 + max_used))

    best_tour = None
    best_len = float('inf')
    best_D_pen = None

    # Get typical edge length for scaling
    positive_edges = edge_distance[edge_distance > 0]
    typical_edge_len = np.median(positive_edges) if len(positive_edges) > 0 else 1.0

    max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
    if max_used < 1e-10:
        max_used = 1.0

    # Compute directed improvements for each tour edge
    improvements = _compute_edge_improvements(edge_distance, tour)
    max_improvement = improvements.max() if improvements.size > 0 else 1.0
    if max_improvement < 1e-10:
        max_improvement = 1.0

    # Build directed penalty: penalize edges proportionally to their improvement potential
    # This guides 2-opt away from edges that are most "costly" to keep
    directed_penalty = np.zeros((n, n))
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        p = improvements[i] / max_improvement
        directed_penalty[a, b] = p
        directed_penalty[b, a] = p

    # Usage-based penalty
    usage = edge_n_used / max_used
    usage_penalty = np.zeros((n, n))
    usage_penalty = usage  # symmetric already

    # Combine: directed + usage, with various weights
    base_alpha_usage = 0.1 + 0.2 * min(1.0, max_used / 10.0)
    base_alpha_directed = 0.15 + 0.25 * min(1.0, max_improvement / typical_edge_len)

    # Strategy: (w_usage, w_directed, decay_usage, decay_directed, scale)
    strategies = [
        (1.0, 0.0, 1.0, 0.0, False),   # pure usage
        (0.0, 1.0, 0.0, 1.0, False),   # pure directed
        (1.0, 1.0, 1.0, 1.0, False),   # combined
        (1.0, 1.5, 1.0, 1.0, False),   # more directed
        (1.5, 1.0, 1.0, 1.0, False),   # more usage
        (1.0, 1.0, 2.0, 1.0, False),   # squared usage
        (1.0, 1.0, 1.0, 2.0, False),   # squared directed
        (1.0, 1.0, 1.5, 1.5, False),   # intermediate decay both
        (0.5, 1.5, 1.0, 1.0, True),    # directed-scaled
        (1.0, 2.0, 1.0, 1.0, True),    # strong directed-scaled
        (1.5, 0.5, 1.0, 1.0, True),    # usage-scaled
        (0.7, 1.2, 1.5, 1.0, False),   # intermediate
        (1.2, 0.8, 1.0, 1.5, False),   # intermediate
        (1.0, 1.0, 1.0, 1.0, True),    # combined scaled
    ]

    for w_u, w_d, dec_u, dec_d, scale in strategies:
        if time.time() - _start_time > _DEADLINE - 10:
            break

        # Build penalty matrix
        pen = np.zeros((n, n))
        if w_u > 0:
            alpha_u = base_alpha_usage * w_u
            if scale:
                pen += alpha_u * typical_edge_len * (usage ** dec_u)
            else:
                pen += alpha_u * (usage ** dec_u)
        if w_d > 0:
            alpha_d = base_alpha_directed * w_d
            if scale:
                pen += alpha_d * typical_edge_len * (directed_penalty ** dec_d)
            else:
                pen += alpha_d * (directed_penalty ** dec_d)

        D_pen = edge_distance + pen

        # Ensure symmetry
        D_pen = (D_pen + D_pen.T) / 2.0
        np.fill_diagonal(D_pen, 0.0)

        # Ensure finite and non-negative
        D_pen = np.maximum(D_pen, 0.0)
        D_pen = np.nan_to_num(D_pen, nan=0.0, posinf=np.finfo(float).max / 2, neginf=0.0)

        # Run 2-opt on the penalized matrix
        tour_pen = _two_opt(D_pen, tour, max_passes=10)

        # Evaluate on true distances
        true_len = _closed_len(edge_distance, tour_pen)

        if true_len < best_len:
            best_len = true_len
            best_tour = tour_pen
            best_D_pen = D_pen

    if best_D_pen is not None:
        return best_D_pen
    else:
        return edge_distance * (1.0 + 0.1 * edge_n_used / (1.0 + max_used))
# EVOLVE-BLOCK-END
