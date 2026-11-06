import numpy as np
import time

_START_TIME = None

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _deadline_guard(margin=5.0):
    """Return True if we're getting close to the 150s limit."""
    return _get_elapsed() > (150.0 - margin)

def _compute_mst_cost(unvisited_nodes, distance_matrix):
    """Compute MST cost for a set of nodes using Prim's algorithm (vectorized)."""
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n <= 1:
        return 0.0
    
    sub = distance_matrix[np.ix_(unv, unv)].copy()
    np.fill_diagonal(sub, 0.0)
    
    # Prim's algorithm
    in_mst = np.zeros(n, dtype=bool)
    in_mst[0] = True
    min_dist = sub[0].copy()
    
    total_cost = 0.0
    for _ in range(1, n):
        # Find node with min distance not in MST
        candidates = np.where(~in_mst)[0]
        if len(candidates) == 0:
            break
        best_idx = candidates[np.argmin(min_dist[candidates])]
        total_cost += min_dist[best_idx]
        in_mst[best_idx] = True
        # Update min distances
        min_dist = np.minimum(min_dist, sub[best_idx])
    
    return total_cost

def _select_insertion_heuristic(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Score each candidate by estimating the incremental tour cost if it were chosen next.
    Uses MST-based estimation of remaining tour length.
    """
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    
    dm = distance_matrix
    d_cur = dm[current_node, unv]
    d_dest = dm[unv, destination_node]
    
    # For each candidate i, estimate the cost of:
    # 1. Going from current to i
    # 2. Then visiting all remaining nodes (unv without i) and returning to destination
    # We approximate the remaining tour cost using MST of remaining nodes.
    
    # Precompute distances from each unv node to all others
    sub = dm[np.ix_(unv, unv)]
    
    scores = np.zeros(n)
    
    for i in range(n):
        # Cost to reach i from current
        cost_to_i = d_cur[i]
        
        # Cost from i to destination (if it's the last)
        # But we need to visit all other nodes first.
        
        # Remaining nodes after choosing i
        remaining_mask = np.ones(n, dtype=bool)
        remaining_mask[i] = False
        remaining = unv[remaining_mask]
        n_rem = remaining.size
        
        if n_rem == 0:
            # i is the last node before destination
            total_cost = cost_to_i + d_dest[i]
        else:
            # Estimate: cost to go from i to remaining, visit them, return to destination
            # Use MST of remaining nodes as a lower bound on path length
            # Plus connection costs from i to MST and from MST to destination
            
            # MST cost of remaining nodes
            mst_cost = _compute_mst_cost(remaining, dm)
            
            # Cost from i to nearest remaining node
            d_i_to_rem = dm[unv[i], remaining]
            cost_i_to_mst = d_i_to_rem.min()
            
            # Cost from MST to destination: approximate by min distance from any remaining node to dest
            d_rem_to_dest = dm[remaining, destination_node]
            cost_mst_to_dest = d_rem_to_dest.min()
            
            # Rough estimate: path from i through remaining to dest
            # This is a lower bound; actual tour will be longer
            total_cost = cost_to_i + cost_i_to_mst + mst_cost + cost_mst_to_dest
        
        scores[i] = total_cost
    
    # Normalize and pick minimum
    score_span = scores.max() - scores.min()
    if score_span > 1e-12:
        scores_norm = (scores - scores.min()) / score_span
    else:
        scores_norm = np.zeros_like(scores)
    
    idx = int(np.argmin(scores_norm))
    return int(unv[idx])

def _held_karp_next(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Finds the optimal next node for the last k nodes using Held-Karp DP.
    Returns the first node in the optimal sequence.
    """
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    
    # Map unvisited nodes to local indices 0..n-1
    full_mask = (1 << n) - 1
    
    INF = float('inf')
    dp = np.full((1 << n, n), INF, dtype=float)
    parent = np.full((1 << n, n), -1, dtype=int)
    
    # Base cases: start at current_node, go to unv[i]
    for i in range(n):
        dp[1 << i, i] = distance_matrix[current_node, unv[i]]
        
    # Fill DP table
    for mask in range(1 << n):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            current_cost = dp[mask, i]
            if current_cost == INF:
                continue
            
            for j in range(n):
                if mask & (1 << j):
                    continue
                
                new_mask = mask | (1 << j)
                step_cost = distance_matrix[unv[i], unv[j]]
                new_cost = current_cost + step_cost
                
                if new_cost < dp[new_mask, j]:
                    dp[new_mask, j] = new_cost
                    parent[new_mask, j] = i

    # Find the best ending node
    best_cost = INF
    best_last_i = -1
    for i in range(n):
        total_cost = dp[full_mask, i] + distance_matrix[unv[i], destination_node]
        if total_cost < best_cost:
            best_cost = total_cost
            best_last_i = i
            
    if best_last_i == -1:
        return int(unv[0])
        
    # Backtrack to find the first node
    current_mask = full_mask
    current_i = best_last_i
    first_i = -1
    
    while current_mask != 0:
        if current_mask & (current_mask - 1) == 0:
            first_i = current_i
            break
        
        prev_i = parent[current_mask, current_i]
        if prev_i == -1:
            break
        current_mask = current_mask ^ (1 << current_i)
        current_i = prev_i
        
    if first_i == -1:
        first_i = 0
        
    return int(unv[first_i])

def _select_fallback_heuristic(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Simple nearest-neighbor fallback with destination awareness."""
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    
    dm = distance_matrix
    d_cur = dm[current_node, unv]
    d_dest = dm[unv, destination_node]
    
    # Weighted combination
    d_cur_span = d_cur.max() - d_cur.min()
    d_dest_span = d_dest.max() - d_dest.min()
    
    if d_cur_span > 1e-12:
        d_cur_norm = (d_cur - d_cur.min()) / d_cur_span
    else:
        d_cur_norm = np.zeros_like(d_cur)
    
    if d_dest_span > 1e-12:
        d_dest_norm = (d_dest - d_dest.min()) / d_dest_span
    else:
        d_dest_norm = np.zeros_like(d_dest)
    
    score = 0.8 * d_cur_norm + 0.2 * d_dest_norm
    idx = int(np.argmin(score))
    return int(unv[idx])

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    
    # Check deadline
    if _deadline_guard(margin=3.0):
        # Use fast fallback
        return _select_fallback_heuristic(current_node, destination_node, unv, distance_matrix)
    
    # Use exact DP for small remaining sets
    if n <= 12:
        return _held_karp_next(current_node, destination_node, unv, distance_matrix)
    
    # Use insertion cost heuristic for larger sets
    return _select_insertion_heuristic(current_node, destination_node, unv, distance_matrix)
