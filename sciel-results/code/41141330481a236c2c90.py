"""Guided local search penalty rule for TSP with adaptive edge penalties."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 320.0  # Safety margin before 330s kill

def _compute_penalty(edge_distance, local_opt_tour, edge_n_used):
    """Compute adaptive penalties for tour edges based on usage frequency."""
    n = len(local_opt_tour)
    if n == 0:
        return np.zeros_like(edge_distance)
    
    tour = np.asarray(local_opt_tour)
    u = tour
    v = np.roll(tour, -1)
    
    # Get true distances for these tour edges
    current_edge_lengths = edge_distance[u, v]
    avg_edge_len = np.mean(current_edge_lengths)
    
    # Fallback for degenerate cases
    if not np.isfinite(avg_edge_len) or avg_edge_len == 0:
        avg_edge_len = np.mean(edge_distance)
        if not np.isfinite(avg_edge_len) or avg_edge_len == 0:
            avg_edge_len = 0.1
    
    # Get usage counts for current tour edges
    edge_counts = edge_n_used[u, v]
    
    # Normalize counts: use log-scale to smooth the penalty
    max_count = np.max(edge_n_used) if edge_n_used.size > 0 else 1
    if max_count == 0:
        max_count = 1
    
    # Use a smooth penalty function: base * (1 + alpha * log(1 + count) / log(1 + max_count))
    alpha = 3.0
    base_penalty = 0.25 * avg_edge_len
    
    penalty_matrix = np.zeros_like(edge_distance)
    for k in range(n):
        ui = u[k]
        vi = v[k]
        count = edge_counts[k]
        # Smooth normalization using log scale
        norm_count = np.log1p(count) / np.log1p(max_count) if max_count > 0 else 0
        penalty = base_penalty * (1.0 + alpha * norm_count)
        penalty_matrix[ui, vi] += penalty
        penalty_matrix[vi, ui] += penalty
    
    return penalty_matrix

def _two_opt(dist, tour):
    """Perform 2-opt improvement on a tour. Returns improved tour."""
    tour = list(tour)
    n = len(tour)
    if n < 4:
        return tour
    
    best_tour = tour[:]
    best_len = _tour_length(dist, best_tour)
    
    for _pass in range(10):
        improved = False
        for i in range(n - 1):
            if time.perf_counter() > _DEADLINE:
                return best_tour
            a = best_tour[i]
            b = best_tour[i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c = best_tour[j]
                d = best_tour[(j + 1) % n]
                
                # Compute delta
                old_len = dist[a, b] + dist[c, d]
                new_len = dist[a, c] + dist[b, d]
                
                if new_len < old_len - 1e-10:
                    # Reverse segment i+1..j
                    best_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
                    best_len = _tour_length(dist, best_tour)
                    improved = True
                    break
        if not improved:
            break
    
    return best_tour

def _tour_length(dist, tour):
    """Compute the length of a closed tour."""
    if len(tour) < 2:
        return 0.0
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += dist[tour[i], tour[(i + 1) % n]]
    return total

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Adaptive GLS penalty: penalize edges in the current tour based on
    their usage frequency, with a base penalty scaled by edge length.
    Uses log-smoothed normalization for more gradual penalty scaling.
    """
    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()
    
    n = len(local_opt_tour)
    if n == 0:
        return edge_distance.copy()
    
    # Create a copy of the true distance matrix
    penalized_dist = edge_distance.copy()
    
    # Compute and apply penalties
    penalty_matrix = _compute_penalty(edge_distance, local_opt_tour, edge_n_used)
    penalized_dist += penalty_matrix
    
    # Ensure the diagonal is 0
    np.fill_diagonal(penalized_dist, 0.0)
    
    # Ensure symmetry
    penalized_dist = (penalized_dist + penalized_dist.T) / 2.0
    
    # Check for finiteness
    if not np.all(np.isfinite(penalized_dist)):
        penalized_dist = np.nan_to_num(penalized_dist, nan=0.0, posinf=np.max(edge_distance), neginf=0.0)
    
    return penalized_dist
# EVOLVE-BLOCK-END
