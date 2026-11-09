"""Guided local search penalty rule for TSP with 100 points."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 330.0
_start_time = None


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Edge-reversal heuristic for guided local search.
    
    For each edge in the current tour, identify the cheapest non-tour edge
    that could replace it (via a 2-opt move), and apply strong penalties to
    the current tour edges to force 2-opt to find these specific replacements.
    
    The key insight: by heavily penalizing tour edges that have good 2-opt
    replacements, we guide the subsequent 2-opt to explore precisely those
    alternative connections, effectively performing a directed perturbation.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Safety margin: return unmodified matrix
        return edge_distance.copy()
    
    n = edge_distance.shape[0]
    D = edge_distance.astype(np.float64).copy()
    tour = list(local_opt_tour)
    tour_len = len(tour)
    
    if tour_len < 4:
        return D
    
    # Compute reference scale: median of off-diagonal distances
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_diag = D[off_diag_mask]
    scale = np.median(off_diag) if len(off_diag) > 0 else 1.0
    if scale < 1e-10:
        scale = 1.0
    
    # Build tour edge mask (boolean for speed)
    tour_edge_mask = np.zeros((n, n), dtype=bool)
    for i in range(tour_len):
        u = tour[i]
        v = tour[(i + 1) % tour_len]
        tour_edge_mask[u, v] = True
        tour_edge_mask[v, u] = True
    
    # Usage statistics
    max_usage = edge_n_used.max() if edge_n_used.size > 0 else 0
    if max_usage < 1e-10:
        max_usage = 1.0
    usage_norm = edge_n_used.astype(np.float64) / max_usage
    
    # Precompute neighbor orderings once
    neighbor_order = np.argsort(D, axis=1)
    
    # Determine search depth based on available time
    check_time = time.time() - _start_time
    remaining = _DEADLINE - 10.0 - check_time
    if remaining < 5.0:
        K = 5
    elif remaining < 20.0:
        K = 10
    else:
        K = min(25, n - 1)
    
    # Edge-reversal: for each tour edge, find best 2-opt replacement
    # and apply a strong penalty to the original edge
    reversal_penalty = np.zeros((n, n), dtype=np.float64)
    
    for i in range(tour_len):
        u = tour[i]
        v = tour[(i + 1) % tour_len]
        
        pred_u = tour[(i - 1) % tour_len]
        succ_v = tour[(i + 2) % tour_len]
        
        best_delta = np.inf
        
        for c_idx in range(K):
            c = neighbor_order[u, c_idx]
            if c == v or c == pred_u:
                continue
            
            for d_idx in range(K):
                d = neighbor_order[v, d_idx]
                if d == u or d == succ_v or d == c:
                    continue
                
                # Check if (c,d) is a tour edge
                if tour_edge_mask[c, d]:
                    continue
                
                # Two 2-opt moves
                delta1 = D[u, c] + D[v, d] - D[u, v] - D[c, d]
                delta2 = D[u, d] + D[v, c] - D[u, v] - D[c, d]
                delta = min(delta1, delta2)
                
                if delta < best_delta:
                    best_delta = delta
        
        # If a negative delta exists, this edge has a good 2-opt replacement
        # Apply a strong penalty proportional to the improvement
        if best_delta < -1e-10:
            # Strong penalty: make this edge very unattractive to 2-opt
            # The penalty should be large enough that 2-opt will prefer
            # to break this edge and try the replacement
            strength = min(-best_delta / scale, 2.0) * scale * 0.5
            reversal_penalty[u, v] = strength
            reversal_penalty[v, u] = strength
    
    # Component 1: Base penalty on all tour edges (mild, to encourage exploration)
    # Scale by usage for finer granularity
    base_scale = scale * 0.05
    adaptive_mult = 1.0 + 0.5 * (usage_norm[tour_edge_mask].mean() if tour_edge_mask.any() else 0.0)
    tour_penalty = base_scale * adaptive_mult * usage_norm * tour_edge_mask
    
    # Component 2: Mild diversification penalty on frequently used edges
    div_penalty = base_scale * 0.3 * usage_norm
    
    # Combine penalties
    penalty = tour_penalty + div_penalty + reversal_penalty
    
    # Ensure symmetry
    penalty = (penalty + penalty.T) / 2.0
    np.fill_diagonal(penalty, 0.0)
    
    # Apply penalty
    D_out = D + penalty
    
    # Ensure finiteness and symmetry
    D_out = np.nan_to_num(D_out, nan=0.0, posinf=scale * 10, neginf=0.0)
    D_out = (D_out + D_out.T) / 2.0
    np.fill_diagonal(D_out, 0.0)
    
    return D_out
# EVOLVE-BLOCK-END
