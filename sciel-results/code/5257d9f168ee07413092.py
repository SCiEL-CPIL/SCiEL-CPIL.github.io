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
    
    # Try multiple penalty strategies and keep the best result
    best_tour = None
    best_len = float('inf')
    best_D_pen = None
    
    # Penalty strategies to try
    # Each strategy: (base_multiplier, use_decay, extra)
    strategies = [
        # (base_multiplier, use_decay, extra)
        (0.1, 1.0, 0),    # mild penalty
        (0.2, 1.0, 0),    # moderate penalty
        (0.3, 1.0, 0),    # stronger penalty
        (0.5, 1.0, 0),    # strong penalty
        (0.1, 2.0, 0),    # mild with squared decay
        (0.2, 2.0, 0),    # moderate with squared decay
        (0.3, 2.0, 0),    # stronger with squared decay
        (0.15, 1.5, 0),   # intermediate
        (0.25, 1.5, 0),   # intermediate
        (0.35, 1.5, 0),   # intermediate
    ]
    
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
    if max_used < 1e-10:
        max_used = 1.0
    
    for base_mult, decay, _ in strategies:
        # Check deadline
        if time.time() - _start_time > _DEADLINE - 10:
            break
        
        # Create penalty matrix
        # penalty[i,j] = base_mult * (edge_n_used[i,j] / max_used) ** decay
        # But only apply to edges not in the current tour? No, apply to all used edges.
        # Actually, guided local search penalizes edges that are in the current local optimum.
        # The edge_n_used tracks how many times each edge has been used.
        
        # Standard GLS: penalize edges in the current tour based on usage count
        # penalty = base * (usage_count / max_usage)
        # Modified distance = original + penalty for edges in current tour
        
        # Let's compute the penalty for edges in the current tour
        D_pen = edge_distance.copy()
        
        # Apply penalty to edges that have been used
        # The penalty is proportional to usage count
        usage = edge_n_used / max_used
        # Apply penalty: D_pen[i,j] += base_mult * usage[i,j] ** decay
        # But we should only penalize edges that are "bad" (frequently used)
        
        # Actually, in GLS, we penalize the edges of the current local optimum.
        # The edge_n_used matrix tracks cumulative usage.
        # A common approach: penalty = alpha * (n_used[i,j] / max_n_used)
        # where alpha is the penalty coefficient.
        
        # Let's try: D_pen[i,j] = D[i,j] + base_mult * (edge_n_used[i,j] / max_used) ** decay
        # This penalizes frequently used edges more.
        
        penalty = base_mult * (usage ** decay)
        D_pen = edge_distance + penalty
        
        # Ensure symmetry
        D_pen = (D_pen + D_pen.T) / 2.0
        np.fill_diagonal(D_pen, 0.0)
        
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
