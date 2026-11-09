"""Guided local search penalty for TSP; try multiple penalty candidates, keep best."""
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


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = edge_distance.shape[0]
    tour = list(local_opt_tour)
    
    # Check if we're close to the deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5:
        # Return a simple penalty to be safe
        max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
        return edge_distance * (1.0 + 0.1 * edge_n_used / (1.0 + max_used))
    
    # Identify edges in the current local optimal tour
    current_tour_edges = np.zeros_like(edge_distance, dtype=bool)
    for i in range(n):
        u, v = tour[i], tour[(i + 1) % n]
        current_tour_edges[u, v] = True
        current_tour_edges[v, u] = True

    max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
    if max_used < 1e-10:
        max_used = 1.0
    
    usage = edge_n_used / max_used
    # usage is normalized [0, 1]
    
    # Generate candidate penalty matrices
    # We will store (D_candidate, description)
    candidates = []
    
    # Base multipliers for penalties
    # We explore a range of strengths
    strengths = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    
    # Type 1: Global Usage Penalty (Additive)
    # D' = D + s * usage^p
    for s in strengths:
        for p in [1.0, 2.0]:
            pen = s * (usage ** p)
            D_cand = edge_distance + pen
            D_cand = (D_cand + D_cand.T) / 2.0
            np.fill_diagonal(D_cand, 0.0)
            candidates.append(D_cand)
            
    # Type 2: Current Tour Edge Penalty (Standard GLS)
    # D' = D + s * current_tour_edges
    for s in strengths:
        pen = np.zeros_like(edge_distance)
        pen[current_tour_edges] = s
        D_cand = edge_distance + pen
        D_cand = (D_cand + D_cand.T) / 2.0
        np.fill_diagonal(D_cand, 0.0)
        candidates.append(D_cand)
        
    # Type 3: Hybrid (Global + Current)
    # D' = D + s1 * usage + s2 * current_tour_edges
    for s1 in [0.1, 0.2]:
        for s2 in [0.2, 0.5, 1.0]:
            pen = s1 * usage
            pen[current_tour_edges] += s2
            D_cand = edge_distance + pen
            D_cand = (D_cand + D_cand.T) / 2.0
            np.fill_diagonal(D_cand, 0.0)
            candidates.append(D_cand)

    # Type 4: Multiplicative Penalty on Used Edges
    # D' = D * (1 + s * usage)
    for s in [0.5, 1.0, 2.0]:
        factor = 1.0 + s * usage
        D_cand = edge_distance * factor
        D_cand = (D_cand + D_cand.T) / 2.0
        np.fill_diagonal(D_cand, 0.0)
        candidates.append(D_cand)

    # Evaluate candidates
    best_tour = None
    best_len = float('inf')
    best_D_pen = None
    
    for D_cand in candidates:
        # Check deadline
        if time.time() - _start_time > _DEADLINE - 10:
            break
        
        # Run 2-opt on the penalized matrix
        # We start from the current tour to see if the penalty breaks it effectively
        tour_pen = _two_opt(D_cand, tour, max_passes=10)
        
        # Evaluate on true distances
        true_len = _closed_len(edge_distance, tour_pen)
        
        if true_len < best_len:
            best_len = true_len
            best_tour = tour_pen
            best_D_pen = D_cand
    
    if best_D_pen is not None:
        return best_D_pen
    else:
        # Fallback: return original distance or a very mild penalty
        return edge_distance.copy()
# EVOLVE-BLOCK-END
