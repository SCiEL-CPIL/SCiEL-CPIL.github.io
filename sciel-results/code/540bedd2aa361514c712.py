"""Guided local search penalty for TSP; try multiple penalty candidates, keep best."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: kill after 330s wall-clock, use 300s as safety margin
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
    
    # Try multiple penalty strategies and keep the best result
    best_tour = None
    best_len = float('inf')
    best_D_pen = None
    
    # Get typical edge length for scaling
    positive_edges = edge_distance[edge_distance > 0]
    typical_edge_len = np.median(positive_edges) if len(positive_edges) > 0 else 1.0
    
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
    if max_used < 1e-10:
        max_used = 1.0
    
    # Adaptive alpha based on usage frequency (from inspiration 2)
    base_alpha = 0.1 + 0.2 * min(1.0, max_used / 10.0)
    
    # Penalty strategies to try: (multiplier, decay, scale_by_typical)
    # multiplier scales the base_alpha, decay affects how usage is weighted
    strategies = [
        (0.5, 1.0, False),    # mild
        (1.0, 1.0, False),    # moderate
        (1.5, 1.0, False),    # stronger
        (2.0, 1.0, False),    # strong
        (1.0, 2.0, False),    # moderate with squared usage
        (1.5, 2.0, False),    # stronger with squared usage
        (1.0, 1.0, True),     # moderate scaled by typical edge
        (1.5, 1.0, True),     # stronger scaled by typical edge
        (2.0, 1.0, True),     # strong scaled by typical edge
        (0.7, 1.5, False),    # intermediate decay
        (1.2, 1.5, False),    # intermediate decay
        (1.8, 1.5, False),    # intermediate decay
    ]
    
    for mult, decay, scale_by_typical in strategies:
        # Check deadline
        if time.time() - _start_time > _DEADLINE - 10:
            break
        
        # Create penalty matrix
        usage = edge_n_used / max_used
        alpha = base_alpha * mult
        if scale_by_typical:
            penalty = alpha * typical_edge_len * (usage ** decay)
        else:
            penalty = alpha * (usage ** decay)
        
        D_pen = edge_distance + penalty
        
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
        # Fallback
        return edge_distance * (1.0 + 0.1 * edge_n_used / (1.0 + max_used))
# EVOLVE-BLOCK-END
