"""Guided local search penalty rule for the traveling salesman problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Strategy:
    1. Penalize edges in the current local-opt tour to force escape.
    2. Scale penalty by cumulative usage (edge_n_used) to target
       frequently-reused edges more strongly.
    3. Use a smooth, bounded penalty that preserves metric-like properties
       enough for 2-opt to work effectively.
    
    Improvements:
    - Nonlinear (quadratic) usage penalty to more aggressively target
      edges that keep getting reused across iterations.
    - Slightly higher base penalty for stronger escape pressure.
    - Scaled usage boost that grows gently with max usage count.
    - Small penalty on "near-tour" edges (edges sharing one endpoint
      with a tour edge) to encourage 2-opt to consider broader swaps.
    """
    n = edge_distance.shape[0]
    
    # Time guard: ensure we finish well within 300s
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Build mask of edges in the current local-opt tour
    tour_mask = np.zeros((n, n), dtype=np.float64)
    # Also build a "near-tour" mask: edges that share exactly one endpoint
    # with a tour edge but are not themselves tour edges
    near_mask = np.zeros((n, n), dtype=np.float64)
    
    # Collect tour edges
    tour_edges = set()
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_mask[a, b] = 1.0
        tour_mask[b, a] = 1.0
        tour_edges.add((min(a, b), max(a, b)))
    
    # Build near-tour mask: for each tour edge (a,b), mark (a,c) and (b,c)
    # for all c that are not a or b and (a,c) is not a tour edge
    # To be efficient, we only consider a few near-neighbors
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        # For endpoint a: mark edges from a to nodes adjacent to b in the tour
        prev_b = local_opt_tour[(i - 1) % n]
        next_b = local_opt_tour[(i + 2) % n]
        for c in (prev_b, next_b):
            if c != a and c != b:
                edge_key = (min(a, c), max(a, c))
                if edge_key not in tour_edges:
                    near_mask[a, c] = 1.0
                    near_mask[c, a] = 1.0
        # For endpoint b: mark edges from b to nodes adjacent to a in the tour
        prev_a = local_opt_tour[(i - 1) % n]
        next_a = local_opt_tour[(i + 2) % n]
        for c in (prev_a, next_a):
            if c != a and c != b:
                edge_key = (min(b, c), max(b, c))
                if edge_key not in tour_edges:
                    near_mask[b, c] = 1.0
                    near_mask[c, b] = 1.0
    
    # Normalize usage with a saturating function to avoid extreme values
    max_used = float(np.max(edge_n_used))
    if max_used > 0:
        # Use a smooth saturating function: usage / (1 + usage)
        # This gives values in (0, 1) and saturates for large usage
        normalized_usage = edge_n_used / (1.0 + edge_n_used)
        # Also compute a quadratic version for stronger penalty on
        # frequently-used edges
        quadratic_usage = (normalized_usage ** 2)
    else:
        normalized_usage = np.zeros_like(edge_n_used)
        quadratic_usage = np.zeros_like(edge_n_used)
    
    # Penalty design:
    # - Base penalty for all current tour edges: ensures escape
    # - Usage-dependent boost: edges used more often get more penalty
    # - Quadratic term to more aggressively target repeatedly-used edges
    # - Small penalty on near-tour edges to encourage broader exploration
    
    # Adaptive base penalty: start moderate, increase slightly with usage
    if max_used > 0:
        # Base penalty grows with usage to encourage stronger escape later
        base_penalty = 0.10 + 0.012 * min(max_used, 20.0)
        # Linear usage boost
        usage_boost = 0.18
        # Quadratic usage boost (stronger for very frequently used edges)
        quad_boost = 0.08 + 0.005 * min(max_used, 20.0)
    else:
        base_penalty = 0.10
        usage_boost = 0.18
        quad_boost = 0.08
    
    # Near-tour penalty (small, to encourage broader 2-opt swaps)
    near_penalty = 0.02 + 0.002 * min(max_used, 10.0)
    
    # Compute penalty for tour edges
    tour_penalty = (base_penalty + usage_boost * normalized_usage + quad_boost * quadratic_usage) * tour_mask
    
    # Compute penalty for near-tour edges
    near_pen = near_penalty * near_mask * (1.0 - tour_mask)
    
    # Combine penalties
    total_penalty = tour_penalty + near_pen
    
    # Apply penalty multiplicatively
    result = edge_distance * (1.0 + total_penalty)
    
    # Ensure finite values and positive distances
    result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=0.0)
    
    # Ensure diagonal is zero (distance from node to itself)
    np.fill_diagonal(result, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
