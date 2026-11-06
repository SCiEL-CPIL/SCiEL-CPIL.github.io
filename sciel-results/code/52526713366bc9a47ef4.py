"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = None
_TIME_LIMIT = 140.0  # seconds, leaving 10s safety margin for 150s limit


def _get_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME


def _time_exceeded():
    return (time.time() - _get_start_time()) > _TIME_LIMIT


def _held_karp_small(candidates, dist_matrix, start_node, end_node, n_cand):
    """
    Exact Held-Karp DP on a small set of candidate nodes to find the optimal
    path from start_node through all candidates to end_node.
    
    candidates: list of node indices (size n_cand)
    dist_matrix: full distance matrix
    start_node: index of current node (not in candidates)
    end_node: index of destination node (not in candidates)
    
    Returns: (best_next_node, best_total_cost)
    """
    if n_cand <= 1:
        return candidates[0], dist_matrix[start_node, candidates[0]] + dist_matrix[candidates[0], end_node]
    
    if n_cand <= 2:
        c0, c1 = candidates
        cost0 = dist_matrix[start_node, c0] + dist_matrix[c0, c1] + dist_matrix[c1, end_node]
        cost1 = dist_matrix[start_node, c1] + dist_matrix[c1, c0] + dist_matrix[c0, end_node]
        if cost0 <= cost1:
            return c0, cost0
        else:
            return c1, cost1
    
    # Map candidates to 0..n_cand-1
    cand_index = {c: i for i, c in enumerate(candidates)}
    
    # dp[mask][i] = min cost to start at start_node, visit exactly the set of candidates
    # represented by mask, and end at candidate i (where i is index in candidates)
    full_mask = (1 << n_cand) - 1
    INF = float('inf')
    
    # dp[mask][i] for all masks and i
    dp = [[INF] * n_cand for _ in range(1 << n_cand)]
    
    # Base case: start from start_node, go to each candidate
    for i in range(n_cand):
        dp[1 << i][i] = dist_matrix[start_node, candidates[i]]
    
    # Fill DP table
    for mask in range(1, 1 << n_cand):
        for i in range(n_cand):
            if not (mask & (1 << i)):
                continue
            if dp[mask][i] == INF:
                continue
            # Try adding each candidate j not yet in mask
            for j in range(n_cand):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                new_cost = dp[mask][i] + dist_matrix[candidates[i], candidates[j]]
                if new_cost < dp[new_mask][j]:
                    dp[new_mask][j] = new_cost
    
    # Find best completion: from any last candidate i, go to end_node
    best_cost = INF
    best_last = -1
    for i in range(n_cand):
        total = dp[full_mask][i] + dist_matrix[candidates[i], end_node]
        if total < best_cost:
            best_cost = total
            best_last = i
    
    # Now backtrack to find the first node after start_node
    # We need to reconstruct the path. Since we only need the NEXT node (first in the path),
    # we can trace back from best_last.
    
    # Actually, we need the first node visited after start_node.
    # Let's store parent pointers.
    
    # Redo with parent tracking
    dp2 = [[INF] * n_cand for _ in range(1 << n_cand)]
    parent = [[-1] * n_cand for _ in range(1 << n_cand)]
    
    for i in range(n_cand):
        dp2[1 << i][i] = dist_matrix[start_node, candidates[i]]
    
    for mask in range(1, 1 << n_cand):
        for i in range(n_cand):
            if not (mask & (1 << i)):
                continue
            if dp2[mask][i] == INF:
                continue
            for j in range(n_cand):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                new_cost = dp2[mask][i] + dist_matrix[candidates[i], candidates[j]]
                if new_cost < dp2[new_mask][j]:
                    dp2[new_mask][j] = new_cost
                    parent[new_mask][j] = i
    
    # Backtrack from best_last at full_mask to find the first node
    mask = full_mask
    i = best_last
    while parent[mask][i] != -1:
        i = parent[mask][i]
        mask = mask ^ (1 << i)
    
    # i is now the first candidate visited
    return candidates[i], best_cost


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city using a combination of nearest-neighbor heuristic and
    exact Held-Karp DP on top-K candidates for better lookahead.
    """
    if len(unvisited_nodes) == 0:
        return current_node  # Should not happen per contract
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Deadline check
    if _time_exceeded():
        # Fall back to nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    # Compute distances from current_node to all unvisited nodes
    dists = distance_matrix[current_node, unvisited_nodes]
    
    # Also compute distances from all unvisited nodes to destination_node
    dists_to_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Score each candidate: immediate cost + cost to destination (2-step lookahead)
    # But we want to use exact DP on top-K for better results
    
    # Select top-K candidates based on a composite score
    # Use immediate distance as primary, but consider destination proximity
    # Sort by immediate distance
    sorted_indices = np.argsort(dists)
    
    # Adaptive K based on remaining nodes
    if n_unvisited <= 12:
        K = n_unvisited
    elif n_unvisited <= 20:
        K = 12
    elif n_unvisited <= 30:
        K = 10
    else:
        K = 8
    
    # Take top-K candidates
    top_k_indices = sorted_indices[:K]
    candidates = unvisited_nodes[top_k_indices]
    
    # If K is small enough, use exact Held-Karp
    if K <= 12:
        try:
            best_next, _ = _held_karp_small(
                list(candidates), distance_matrix, current_node, destination_node, K
            )
            return int(best_next)
        except Exception:
            pass
    
    # Fallback: use 2-step lookahead (nearest neighbor considering destination)
    # Score: dist(current, i) + dist(i, destination)
    scores = dists[top_k_indices] + dists_to_dest[top_k_indices]
    best_local_idx = np.argmin(scores)
    return int(candidates[best_local_idx])
# EVOLVE-BLOCK-END
