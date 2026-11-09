"""Guided local search penalty rule for TSP with bootstrap-tuned adaptive GLS penalty."""
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 320.0  # Safety margin against 330s kill


def _t2opt(dist, tour, max_passes=10):
    """Run 2-opt on the given tour under the given distance matrix."""
    n = len(tour)
    best_tour = list(tour)
    best_len = 0.0
    for i in range(n):
        a = best_tour[i]
        b = best_tour[(i + 1) % n]
        best_len += dist[a, b]
    
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            a = best_tour[i]
            b = best_tour[i + 1]
            for j in range(i + 2, n):
                if j == (i + n - 1) % n:
                    continue
                c = best_tour[j]
                d = best_tour[(j + 1) % n]
                delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
                if delta < -1e-10:
                    best_tour[i + 1:j + 1] = best_tour[i + 1:j + 1][::-1]
                    best_len += delta
                    improved = True
        if not improved:
            break
    return best_tour, best_len


def _nn_tour(dist):
    """Build a nearest-neighbour tour."""
    n = len(dist)
    unvisited = set(range(n))
    unvisited.remove(0)
    tour = [0]
    while unvisited:
        cur = tour[-1]
        nxt = min(unvisited, key=lambda v: dist[cur, v])
        tour.append(nxt)
        unvisited.remove(nxt)
    return tour


def _bootstrap(edge_distance, local_opt_tour, edge_n_used, n_candidates=12):
    """
    Try several penalty scales, run 2-opt on each, and pick the alpha
    that gives the best true-distance tour length after 2-opt.
    """
    n = len(local_opt_tour)
    
    # Compute tour length and edge usage sum
    tour_len = 0.0
    edge_sum = 0.0
    for i in range(n):
        j = (i + 1) % n
        tour_len += edge_distance[i, j]
        edge_sum += edge_n_used[i, j]
    
    if edge_sum <= 0:
        return edge_distance.copy()
    
    # Candidate alphas: geometric spread around a base scale
    base = 0.15 * tour_len / edge_sum
    # Try a range of multipliers to find the sweet spot
    multipliers = np.array([0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0])
    
    best_alpha = 1.0 * base
    best_true_len = float('inf')
    best_tour = list(local_opt_tour)
    
    # Also include a small set of fixed absolute alphas for robustness
    fixed_alphas = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    all_alphas = list(multipliers * base) + fixed_alphas
    # Deduplicate and sort
    all_alphas = sorted(set(round(a, 10) for a in all_alphas if 0 < a < 1.0))
    if not all_alphas:
        all_alphas = [base]
    
    for alpha in all_alphas:
        if time.perf_counter() > _DEADLINE - 30:
            break
        
        # Build penalized matrix
        penalty = np.zeros((n, n))
        for i in range(n):
            j = (i + 1) % n
            penalty[i, j] = alpha * edge_n_used[i, j]
            penalty[j, i] = alpha * edge_n_used[j, i]
        
        penalized = edge_distance + penalty
        penalized = (penalized + penalized.T) / 2.0
        np.fill_diagonal(penalized, 0.0)
        
        # Run 2-opt on penalized matrix
        tour_p, _ = _t2opt(penalized, local_opt_tour, max_passes=10)
        
        # Evaluate true length
        true_len = 0.0
        for i in range(n):
            a = tour_p[i]
            b = tour_p[(i + 1) % n]
            true_len += edge_distance[a, b]
        
        if true_len < best_true_len:
            best_true_len = true_len
            best_alpha = alpha
            best_tour = tour_p
    
    # Build final penalized matrix with best alpha
    penalty = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        penalty[i, j] = best_alpha * edge_n_used[i, j]
        penalty[j, i] = best_alpha * edge_n_used[j, i]
    
    result = edge_distance + penalty
    result = (result + result.T) / 2.0
    np.fill_diagonal(result, 0.0)
    
    # Tiny noise for symmetry breaking
    rng = np.random.default_rng(42)
    noise = rng.uniform(0, 1e-6, size=(n, n))
    result += noise
    result = (result + result.T) / 2.0
    np.fill_diagonal(result, 0.0)
    result = np.maximum(result, 0.0)
    result = np.where(np.isfinite(result), result, edge_distance)
    
    return result


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Adaptive GLS penalty with bootstrap-tuned scale.
    """
    n = len(local_opt_tour)
    
    if time.perf_counter() > _DEADLINE - 20:
        # Emergency: return unpenalized
        return edge_distance.copy()
    
    # Fast path: if no edges have been used yet, no penalty needed
    edge_sum = 0.0
    for i in range(n):
        j = (i + 1) % n
        edge_sum += edge_n_used[i, j]
    
    if edge_sum <= 0:
        return edge_distance.copy()
    
    return _bootstrap(edge_distance, local_opt_tour, edge_n_used)
