"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache to avoid recomputing MST-based heuristics repeatedly
_mst_cache = {}
_call_count = 0
_start_time = None


def _compute_mst_residual(unvisited_nodes, distance_matrix):
    """
    Compute a lower-bound estimate of the remaining tour cost using MST.
    For a set of unvisited nodes, the MST cost is a lower bound on the
    cost to connect them. We use this as a residual estimate.
    
    To make this efficient and useful for selection, we compute the MST
    weight of the subgraph induced by unvisited nodes, and also consider
    the connection cost from current/destination to this subgraph.
    """
    n = len(unvisited_nodes)
    if n <= 1:
        return 0.0
    
    # Get the submatrix for unvisited nodes
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Use Prim's algorithm for MST
    in_mst = np.zeros(n, dtype=bool)
    min_edge = np.full(n, np.inf)
    min_edge[0] = 0.0
    mst_weight = 0.0
    
    for _ in range(n):
        # Find the node with the smallest edge not yet in MST
        # Use argmin on min_edge, masking out already-included nodes
        min_edge[in_mst] = np.inf
        u = int(np.argmin(min_edge))
        if min_edge[u] == np.inf:
            break  # Disconnected, shouldn't happen
        in_mst[u] = True
        mst_weight += min_edge[u]
        # Update min_edge for neighbors
        for v in range(n):
            if not in_mst[v] and sub_dist[u, v] < min_edge[v]:
                min_edge[v] = sub_dist[u, v]
    
    return mst_weight


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in a TSP tour construction.
    
    Uses a combined heuristic:
    - Immediate distance from current_node to candidate
    - A residual estimate based on MST of remaining unvisited nodes after visiting candidate
    - Distance from candidate to destination (to help close the tour)
    
    The score for each candidate is:
      w1 * dist(current, cand) + w2 * dist(cand, dest) + w3 * mst_residual_after_cand
    
    where mst_residual_after_cand estimates the cost to complete the tour
    from the remaining unvisited nodes.
    """
    global _mst_cache, _call_count, _start_time
    
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard: if we've been running too long, fall back to nearest neighbor
    elapsed = time.time() - _start_time
    if elapsed > 110:  # Safety margin before 120s limit
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    unvisited = np.asarray(unvisited_nodes)
    n_unvisited = len(unvisited)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited[0])
    
    # Compute distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited]
    # Compute distances from destination to all unvisited
    dist_to_dest = distance_matrix[destination_node, unvisited]
    
    # For each candidate, compute the MST residual of the remaining nodes
    # This is expensive, so we use a cheaper approximation for larger sets
    if n_unvisited <= 30:
        # Compute MST for each candidate's remaining set
        mst_residuals = np.zeros(n_unvisited)
        for i in range(n_unvisited):
            # Remaining nodes after visiting candidate i
            remaining = np.delete(unvisited, i)
            if len(remaining) <= 1:
                mst_residuals[i] = 0.0
            else:
                # Cache key: sorted tuple of remaining nodes
                key = tuple(sorted(remaining))
                if key in _mst_cache:
                    mst_residuals[i] = _mst_cache[key]
                else:
                    mst_residuals[i] = _compute_mst_residual(remaining, distance_matrix)
                    # Only cache if the set is small enough to avoid memory issues
                    if len(_mst_cache) < 5000:
                        _mst_cache[key] = mst_residuals[i]
    else:
        # For larger sets, use a cheaper approximation:
        # The MST of all unvisited nodes is a rough lower bound
        # We approximate the change by using the minimum edge from candidate to the rest
        all_mst = _compute_mst_residual(unvisited, distance_matrix)
        
        # For each candidate, estimate residual as:
        # all_mst - (some reduction) + connection costs
        # Simpler: use the minimum distance from candidate to any other unvisited
        # as a proxy for how "central" the candidate is
        min_dist_to_others = np.full(n_unvisited, np.inf)
        for i in range(n_unvisited):
            others = np.delete(unvisited, i)
            if len(others) > 0:
                min_dist_to_others[i] = np.min(distance_matrix[unvisited[i], others])
        
        # Normalize min_dist_to_others to [0, 1]
        min_d = np.min(min_dist_to_others)
        max_d = np.max(min_dist_to_others)
        if max_d > min_d:
            normalized_central = (min_dist_to_others - min_d) / (max_d - min_d)
        else:
            normalized_central = np.zeros(n_unvisited)
        
        # Use this as a rough residual estimate
        mst_residuals = normalized_central * all_mst
    
    # Combine the scores
    # We want to minimize: immediate cost + residual + connection to destination
    # Weight the components
    w_immediate = 1.0
    w_residual = 0.5
    w_dest = 0.3
    
    scores = (w_immediate * dist_from_current + 
              w_residual * mst_residuals + 
              w_dest * dist_to_dest)
    
    # Select the candidate with the minimum score
    best_idx = int(np.argmin(scores))
    return int(unvisited[best_idx])

# EVOLVE-BLOCK-END
