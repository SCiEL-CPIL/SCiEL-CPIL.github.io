"""Guided local search penalty rule for TSP with 100 points."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 330.0
_start_time = None


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Applies calibrated additive penalties to encourage 2-opt to escape
    local optima. The penalty structure:
    1. Strong penalty on edges in the current local-opt tour (scaled by usage)
    2. Mild penalty on all frequently-used edges (diversification)
    3. Adaptive calibration based on tour quality vs. theoretical lower bound
    
    Returns a modified distance matrix that biases 2-opt toward unexplored
    edge combinations.
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
    
    # Build tour edge mask
    tour_edge_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(tour_len):
        u = tour[i]
        v = tour[(i + 1) % tour_len]
        tour_edge_mask[u, v] = 1.0
        tour_edge_mask[v, u] = 1.0
    
    # Usage statistics
    max_usage = edge_n_used.max() if edge_n_used.size > 0 else 0
    if max_usage < 1e-10:
        max_usage = 1.0
    usage_norm = edge_n_used.astype(np.float64) / max_usage
    
    # Current tour length
    tour_length = 0.0
    for i in range(tour_len):
        tour_length += D[tour[i], tour[(i + 1) % tour_len]]
    
    # Theoretical lower bound for TSP on n uniform points in unit square:
    # ~0.7 * sqrt(n) (based on Beasley's results)
    lower_bound = 0.7 * np.sqrt(n)
    quality_ratio = tour_length / lower_bound  # > 1 means worse than ideal
    
    # Mean usage of current tour edges
    tour_edge_usages = usage_norm[tour_edge_mask > 0.5]
    mean_tour_usage = tour_edge_usages.mean() if len(tour_edge_usages) > 0 else 0.0
    
    # Adaptive penalty calibration:
    # - Base scale: fraction of typical edge distance
    # - Increase as we iterate more (higher usage) and as tour is further from optimal
    base_scale = scale * 0.12
    adaptive_mult = 1.0 + 0.6 * mean_tour_usage + 0.4 * max(0.0, quality_ratio - 1.0)
    
    # Component 1: Strong penalty on current tour edges
    # Scale by individual edge usage for finer granularity
    tour_penalty = base_scale * adaptive_mult * usage_norm * tour_edge_mask
    
    # Component 2: Mild diversification penalty on all used edges
    div_penalty = base_scale * 0.25 * usage_norm
    
    # Component 3: Edge-reversal bias - for each tour edge, slightly penalize
    # the "neighbor" edges that would be created by a 2-opt swap involving it.
    # This is done by identifying, for each tour edge (u,v), the shortest non-tour
    # edge from u and from v, and adding a small penalty to encourage 2-opt to
    # consider those specific replacements.
    reversal_penalty = np.zeros((n, n), dtype=np.float64)
    
    # For speed, only process a subset of tour edges if n is large
    # and we're running low on time
    check_time = time.time() - _start_time
    if check_time < _DEADLINE - 10.0:
        # For each tour edge, find the best 2-opt partner
        # Use a limited search for speed
        K = min(15, n - 1)
        
        # Precompute neighbor orderings
        neighbor_order = np.argsort(D, axis=1)
        
        for i in range(tour_len):
            u = tour[i]
            v = tour[(i + 1) % tour_len]
            
            # Find best 2-opt move removing edge (u, v)
            # Try top-K neighbors of u and v
            best_delta = np.inf
            best_pair = None
            
            pred_u = tour[(i - 1) % tour_len]
            succ_v = tour[(i + 2) % tour_len]
            
            for c_idx in range(K):
                c = neighbor_order[u, c_idx]
                if c == v or c == pred_u:
                    continue
                
                for d_idx in range(K):
                    d = neighbor_order[v, d_idx]
                    if d == u or d == succ_v or d == c:
                        continue
                    
                    # Check if (c,d) is a tour edge
                    if tour_edge_mask[c, d] > 0.5:
                        continue
                    
                    # Two 2-opt moves
                    delta1 = D[u, c] + D[v, d] - D[u, v] - D[c, d]
                    delta2 = D[u, d] + D[v, c] - D[u, v] - D[c, d]
                    delta = min(delta1, delta2)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_pair = (c, d, delta1, delta2)
            
            # If a good replacement exists, add a small penalty to encourage it
            if best_pair is not None and best_delta < 0:
                c, d, delta1, delta2 = best_pair
                # Penalize the original tour edge slightly more if improvement is good
                strength = min(-best_delta / scale, 1.0) * base_scale * 0.3
                D[u, v] += strength
                D[v, u] += strength
    
    # Combine all penalties
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
