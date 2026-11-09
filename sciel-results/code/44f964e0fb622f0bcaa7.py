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

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a triangle-inequality slack heuristic.
    
    The score for a candidate node c is:
    Slack(c) + lambda * dist(c, destination)
    
    Where Slack(c) = dist(current, c) + dist(c, v_best) - dist(current, v_best)
    and v_best is the unvisited node closest to c.
    
    This geometric proxy encourages smooth, non-crossing tour segments.
    """
    # Time guard
    if _get_time_remaining() < 0.5:
        if len(unvisited_nodes) == 0:
            return int(destination_node)
        # Fallback to nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    candidates = unvisited_nodes
    n = len(candidates)

    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(candidates[0])
    
    if n == 2:
        c0 = candidates[0]
        c1 = candidates[1]
        # Exact 2-step lookahead
        cost0 = distance_matrix[current_node, c0] + distance_matrix[c0, c1] + distance_matrix[c1, destination_node]
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c0] + distance_matrix[c0, destination_node]
        if cost0 < cost1:
            return int(c0)
        else:
            return int(c1)

    # Vectorized Slack Calculation
    cand_idx = candidates.astype(int)
    
    # 1. dist(current, candidates)
    d_curr = distance_matrix[current_node, cand_idx]
    
    # 2. Distance matrix among candidates
    cand_dist = distance_matrix[np.ix_(cand_idx, cand_idx)]
    
    # 3. Find nearest neighbor for each candidate within the unvisited set
    #    Set diagonal to inf to exclude self
    np.fill_diagonal(cand_dist, np.inf)
    
    # v_best_idx is the index in the 'candidates' array of the nearest neighbor
    v_best_idx = np.argmin(cand_dist, axis=1)
    # dist(c, v_best)
    d_c_vbest = np.min(cand_dist, axis=1)
    
    # The actual node indices of the nearest neighbors
    v_best_nodes = cand_idx[v_best_idx]
    
    # 4. dist(current, v_best)
    d_curr_vbest = distance_matrix[current_node, v_best_nodes]
    
    # 5. Calculate Slack
    # Slack = d(curr, c) + d(c, v_best) - d(curr, v_best)
    slack = d_curr + d_c_vbest - d_curr_vbest
    
    # 6. Destination bias
    # Weakened bias: lambda = 0.5 / n
    lambda_w = 0.5 / n
    d_dest = distance_matrix[cand_idx, destination_node]
    
    scores = slack + lambda_w * d_dest
    
    idx = np.argmin(scores)
    return int(candidates[idx])
