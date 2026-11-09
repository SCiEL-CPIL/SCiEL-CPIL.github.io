"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def _get_time_remaining():
    """Return seconds remaining before the 150s limit (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)


def _mst_lower_bound(nodes, distance_matrix):
    """
    Compute a lower bound on the cost to visit all remaining nodes using MST.
    This approximates the minimum additional tour length needed.
    """
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    # Use Prim's algorithm for MST on the subgraph of remaining nodes
    # This gives a lower bound on the path length through these nodes
    in_mst = [False] * n
    min_dist = [float('inf')] * n
    min_dist[0] = 0.0
    total = 0.0
    
    for _ in range(n):
        # Find unvisited node with minimum distance
        u = -1
        best = float('inf')
        for i in range(n):
            if not in_mst[i] and min_dist[i] < best:
                best = min_dist[i]
                u = i
        if u == -1:
            break
        in_mst[u] = True
        total += min_dist[u]
        
        # Update distances
        node_u = nodes[u]
        for v in range(n):
            if not in_mst[v]:
                d = distance_matrix[node_u, nodes[v]]
                if d < min_dist[v]:
                    min_dist[v] = d
    
    return total


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 0.5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    candidates = unvisited_nodes
    n = len(candidates)

    if n == 0:
        return int(destination_node)

    if n == 1:
        return int(candidates[0])

    if n == 2:
        # For 2 nodes, pick the one that minimizes the total closing edge
        d0 = distance_matrix[candidates[0], destination_node]
        d1 = distance_matrix[candidates[1], destination_node]
        if d0 < d1:
            return int(candidates[0])
        else:
            return int(candidates[1])

    # For small n, use a hybrid approach: immediate cost + MST-based global cost
    # The key insight: we want to avoid creating disconnected subtours
    # and ensure the remaining nodes can be visited efficiently
    
    # Score 1: Immediate connection cost (current -> candidate)
    dist_from_current = distance_matrix[current_node, candidates]
    
    # Score 2: Closing edge cost (candidate -> destination)
    dist_to_dest = distance_matrix[candidates, destination_node]
    
    # Score 3: MST-based lower bound for the remaining path
    # For each candidate, compute what the MST cost would be if we
    # removed that candidate from the unvisited set
    # This captures global connectivity
    
    # Adaptive weights based on how many nodes remain
    # When many nodes remain, global structure matters more
    # When few remain, immediate connections matter more
    if n > 20:
        # Early in tour: emphasize global connectivity
        w_immediate = 0.3
        w_closing = 0.2
        w_mst = 0.5
    elif n > 10:
        # Middle: balanced
        w_immediate = 0.4
        w_closing = 0.3
        w_mst = 0.3
    else:
        # Late: emphasize immediate and closing
        w_immediate = 0.5
        w_closing = 0.4
        w_mst = 0.1
    
    # Compute MST-based score for each candidate
    mst_scores = np.zeros(n)
    
    # Optimization: for large n, only compute MST for top-k candidates by immediate cost
    if n > 30:
        # Get top-k candidates by immediate cost
        k = min(10, n)
        top_k_indices = np.argsort(dist_from_current)[:k]
        
        # Compute MST scores only for these candidates
        for i in range(n):
            if i in top_k_indices:
                # Remove candidate i from unvisited, compute MST of remaining
                remaining = np.delete(candidates, i)
                mst_scores[i] = _mst_lower_bound(remaining, distance_matrix)
            else:
                # Penalize non-top-k candidates
                mst_scores[i] = mst_scores[top_k_indices[0]] * 1.5
    else:
        # Compute MST scores for all candidates
        for i in range(n):
            remaining = np.delete(candidates, i)
            mst_scores[i] = _mst_lower_bound(remaining, distance_matrix)
    
    # Normalize scores to [0, 1] range for combination
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-10:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    norm_immediate = normalize(dist_from_current)
    norm_closing = normalize(dist_to_dest)
    norm_mst = normalize(mst_scores)
    
    # Combined score
    scores = w_immediate * norm_immediate + w_closing * norm_closing + w_mst * norm_mst
    
    idx = np.argmin(scores)
    return int(candidates[idx])
# EVOLVE-BLOCK-END
