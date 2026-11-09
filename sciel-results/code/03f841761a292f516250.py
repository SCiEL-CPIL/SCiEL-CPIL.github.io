"""Guided local search penalty rule for TSP with 100 points."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 325.0
_start_time = None


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Edge-reversal heuristic for guided local search.
    
    For each edge in the current tour, identify the cheapest non-tour edge
    that could replace it (via a 2-opt move), and apply penalties to the
    current tour edges to encourage 2-opt to find these replacements.
    
    Returns a modified distance matrix that biases 2-opt away from
    the current tour's edges toward better alternatives.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Safety margin: return unmodified matrix if we're about to timeout
        return edge_distance.copy()
    
    n = edge_distance.shape[0]
    penalized = edge_distance.astype(np.float64).copy()
    
    # Compute reference scale
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_diag = edge_distance[off_diag_mask]
    scale = np.median(off_diag) if len(off_diag) > 0 else 1.0
    if scale < 1e-10:
        scale = 1.0
    
    tour = list(local_opt_tour)
    tour_len = len(tour)
    
    if tour_len < 4:
        return penalized
    
    # Identify tour edges
    tour_edges = set()
    for i in range(tour_len):
        u = tour[i]
        v = tour[(i + 1) % tour_len]
        tour_edges.add((min(u, v), max(u, v)))
    
    # Precompute: for each node, sort neighbors by distance (excluding self)
    # This helps find good candidates quickly
    neighbor_order = []
    for i in range(n):
        row = edge_distance[i].copy()
        row[i] = np.inf
        neighbor_order.append(np.argsort(row))
    
    # Build a set for fast lookup of tour edges
    def is_tour_edge(u, v):
        if u == v:
            return False
        return (min(u, v), max(u, v)) in tour_edges
    
    # For each tour edge, find the best 2-opt move involving it
    # and penalize the tour edge to encourage that move
    penalty_strength = 0.4  # Base multiplier for tour edges
    usage_bonus = 0.03      # Additional penalty per usage
    
    for i in range(tour_len):
        a = tour[i]
        b = tour[(i + 1) % tour_len]
        
        # Find best 2-opt move that removes edge (a, b)
        # Try all pairs (c, d) where c is not adjacent to a in tour, 
        # d is not adjacent to b in tour, and (c,d) is not a tour edge
        best_delta = np.inf
        best_c = -1
        best_d = -1
        
        # Limit search to top-K neighbors for speed
        K = min(25, n - 1)
        
        for c_idx in range(K):
            c = neighbor_order[a][c_idx]
            if c == b:
                continue
            # c should not be the predecessor of a in the tour
            pred_a = tour[(i - 1) % tour_len]
            if c == pred_a:
                continue
            
            for d_idx in range(K):
                d = neighbor_order[b][d_idx]
                if d == a:
                    continue
                # d should not be the successor of b in the tour
                succ_b = tour[(i + 2) % tour_len]
                if d == succ_b:
                    continue
                if c == d:
                    continue
                if is_tour_edge(c, d):
                    continue
                if is_tour_edge(a, c) and is_tour_edge(b, d):
                    continue
                if is_tour_edge(a, d) and is_tour_edge(b, c):
                    continue
                
                # Two possible 2-opt moves:
                # Move 1: remove (a,b) and (c,d), add (a,c) and (b,d)
                delta1 = edge_distance[a, c] + edge_distance[b, d] - edge_distance[a, b] - edge_distance[c, d]
                # Move 2: remove (a,b) and (c,d), add (a,d) and (b,c)
                delta2 = edge_distance[a, d] + edge_distance[b, c] - edge_distance[a, b] - edge_distance[c, d]
                
                delta = min(delta1, delta2)
                if delta < best_delta:
                    best_delta = delta
                    best_c = c
                    best_d = d
        
        # Apply penalty to the tour edge (a, b)
        # Stronger penalty if there's a good replacement available
        if best_delta < 0:
            # Good improvement exists - penalize more strongly
            mult = penalty_strength + 0.6 * min(-best_delta / scale, 1.0)
        else:
            # No good improvement - lighter penalty
            mult = penalty_strength * 0.7
        
        # Add usage-based bonus
        usage = edge_n_used[a, b]
        mult += usage_bonus * usage
        
        penalized[a, b] *= mult
        penalized[b, a] *= mult
    
    # Also add a small global usage penalty to break ties
    max_usage = edge_n_used.max()
    if max_usage > 0:
        usage_norm = edge_n_used.astype(np.float64) / max_usage
        penalized += 0.05 * scale * usage_norm
    
    # Ensure symmetry
    penalized = (penalized + penalized.T) / 2.0
    
    # Ensure finite values
    penalized = np.where(np.isfinite(penalized), penalized, 1e10)
    
    return penalized
# EVOLVE-BLOCK-END
