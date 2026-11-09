"""Guided local search penalty for TSP; Targeted 2-opt escape heuristic."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Applies a targeted penalty to edges in the current tour that are 'traps'.
    
    Strategy:
    1. Identify the current tour edges.
    2. For each edge, simulate a 2-opt removal (reversing the segment).
    3. If the removal does not improve the tour length (i.e., the edge is part of 
       a local minimum structure), mark it as a 'trap'.
    4. Apply a strong penalty to these trap edges to force the subsequent 2-opt 
       to break them, while leaving other edges (especially unused ones) relatively 
       unchanged. This is more targeted than penalizing all used edges based on 
       history, which can distort the global geometry unnecessarily.
    
    Args:
    edge_distance: np.ndarray, true distance matrix.
    local_opt_tour: list or np.ndarray, the current tour as a sequence of node indices.
    edge_n_used: np.ndarray, symmetric matrix of usage counts.
    
    Returns:
    np.ndarray: Modified distance matrix for the next 2-opt phase.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin before 300s limit
    
    D = np.asarray(edge_distance, dtype=float)
    N = np.asarray(edge_n_used, dtype=float)
    tour = np.asarray(local_opt_tour, dtype=int)
    
    # Symmetrize inputs
    D = (D + D.T) / 2.0
    N = (N + N.T) / 2.0
    
    n = len(tour)
    if n < 4:
        return D.copy()
        
    # Ensure the tour is a valid permutation
    # The evaluator passes a valid tour, but we check just in case
    if len(np.unique(tour)) != n:
        return D.copy()
        
    # Calculate current tour length
    tour_len = 0.0
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        tour_len += D[u, v]
        
    # Identify edges in the current tour
    # We will store indices of edges that are "traps"
    # An edge (u, v) is a trap if removing it (and reversing the path between u and v) 
    # does not result in a shorter tour.
    
    # Efficiently check 2-opt moves
    # A 2-opt move removes edges (a, b) and (c, d) and adds (a, c) and (b, d).
    # In terms of tour indices i and j (where i < j), the edges are (tour[i], tour[i+1]) and (tour[j], tour[j+1]).
    # The new edges would be (tour[i], tour[j]) and (tour[i+1], tour[j+1]).
    
    # We want to find edges in the current tour that, if removed, do not improve the tour.
    # Actually, the standard 2-opt check is: 
    # delta = D[a,c] + D[b,d] - D[a,b] - D[c,d]
    # If delta < 0, the move is improving.
    
    # We iterate over all pairs of edges in the tour to see if they are "stable" (delta >= 0).
    # However, we only care about penalizing the edges that are part of the current local optimum.
    # If an edge is involved in ANY improving 2-opt move, it's not a trap for that specific move.
    # But if the tour is a local optimum, NO 2-opt move improves it. 
    # Wait, the input `local_opt_tour` is the result of 2-opt on the *previous* penalized matrix.
    # It might not be a 2-opt local optimum on the *true* distances.
    # The evaluator runs 2-opt on the returned matrix.
    
    # Refined Strategy:
    # The goal is to break the current local optimum found by 2-opt on the *true* distances 
    # (or the previous penalized ones). 
    # If we simply penalize all edges in the current tour, 2-opt will try to remove them.
    # But if we penalize them too much, we might force 2-opt into a bad state.
    
    # Let's try a "soft" targeted penalty.
    # 1. Compute the 2-opt delta for all pairs of edges in the current tour.
    # 2. If the tour is a local optimum on true distances, all deltas are >= 0.
    # 3. We identify the edges that are "most stable" or part of the "core" of the local optimum.
    # 4. Actually, a simpler and robust heuristic: 
    #    Apply a penalty to edges in the current tour proportional to their usage count, 
    #    BUT also add a "breakage" bonus to edges that are NOT in the current tour but connect 
    #    nodes that are far apart in the tour index space (long-range connections), 
    #    to encourage 2-opt to consider large structural changes.
    
    # Let's stick to the "Edge Reversal" idea from the directive but implemented efficiently.
    # For each edge in the tour, if it is not part of any improving 2-opt move on the TRUE distances,
    # it is a "trap" edge. We penalize it.
    
    # Precompute tour edge lengths
    tour_edges = []
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        tour_edges.append((u, v, D[u, v]))
        
    # Check for improving 2-opt moves
    # This is O(n^2) which is 10,000 operations for n=100. Very fast.
    # We want to know which edges are involved in NO improving move.
    # If an edge is not involved in any improving move, it's a candidate for penalty.
    
    # However, if the tour is already a 2-opt local optimum on true distances, 
    # then ALL edges are "traps". In that case, we should penalize all of them.
    # If the tour is NOT a local optimum on true distances (because it was optimized 
    # on a penalized matrix), then some edges might be "good" to keep.
    
    # Let's calculate the delta for all 2-opt moves.
    # We mark an edge as "movable" if it is part of at least one improving 2-opt move.
    # Edges that are "not movable" are the ones holding the tour in a local minimum 
    # (or pseudo-minimum) and should be penalized to force a change.
    
    is_movable = np.zeros(n, dtype=bool)
    
    # Optimization: Only check pairs (i, j) where j > i + 1
    for i in range(n - 1):
        u1 = tour[i]
        v1 = tour[i + 1]
        len1 = D[u1, v1]
        
        # Check time budget occasionally
        if time.time() > deadline:
            break
            
        for j in range(i + 2, n):
            if j == n - 1 and i == 0:
                continue # Skip the wrap-around pair (0, n-1) as it's the same edge pair effectively
            
            u2 = tour[j]
            v2 = tour[(j + 1) % n]
            len2 = D[u2, v2]
            
            # New edges: (u1, u2) and (v1, v2)
            new_len = D[u1, u2] + D[v1, v2]
            
            if new_len < len1 + len2:
                # Improving move found. Mark edges i and j as movable.
                is_movable[i] = True
                is_movable[j] = True
                
    # If the loop broke early due to time, is_movable might be incomplete.
    # In that case, we fall back to a simpler penalty.
    time_elapsed = time.time() - start_time
    if time_elapsed > 5.0:
        # Fallback: Simple additive penalty
        lambda_val = 0.5 / (1.0 + np.sum(N > 0) / 50.0)
        new_dist = D + lambda_val * N
        np.fill_diagonal(new_dist, 0.0)
        return new_dist

    # Now, penalize the edges that are NOT movable.
    # These are the "stable" edges of the current local optimum.
    # We want to force 2-opt to break them.
    
    # Penalty magnitude:
    # We use a strong penalty for stable edges.
    # We also add a small global penalty based on usage to encourage diversity.
    
    new_dist = D.copy()
    
    # Global usage penalty (small)
    # lambda_global = 0.1
    # new_dist += 0.1 * N
    
    # Targeted penalty for stable edges
    # For each edge (i, i+1) in the tour, if not is_movable[i], increase its distance.
    # How much? 
    # If we set it to infinity, 2-opt will definitely break it.
    # But 2-opt might get stuck if ALL edges are broken.
    # Since we only break the "stable" ones, and if the tour was a local optimum, ALL are stable.
    # So we would break ALL edges. This is equivalent to saying "the current tour is bad, find a new one".
    # This is effectively a "kick".
    
    # Let's set the distance of stable edges to a large value, e.g., max_dist * 1.5
    max_dist = np.max(D)
    penalty_val = max_dist * 2.0  # Strong penalty
    
    for i in range(n):
        if not is_movable[i]:
            u = tour[i]
            v = tour[(i + 1) % n]
            new_dist[u, v] = penalty_val
            new_dist[v, u] = penalty_val
            
    # Ensure diagonal is 0
    np.fill_diagonal(new_dist, 0.0)
    
    # Ensure symmetry
    new_dist = (new_dist + new_dist.T) / 2.0
    
    # Safety check
    if not np.all(np.isfinite(new_dist)):
        return D.copy()
        
    return new_dist
# EVOLVE-BLOCK-END
