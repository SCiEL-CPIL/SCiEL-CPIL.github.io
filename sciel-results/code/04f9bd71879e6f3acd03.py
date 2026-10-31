# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_LIMIT = 110.0  # Safety margin before 120s/150s limit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Hybrid minimax + cumulative TSP heuristic.
    
    Structural change: Instead of pure sum-based lookahead, we combine:
    1. Cumulative cost: d(cur, c) + min_{j!=i}[d(c,j) + gamma*d(j,dest)]
    2. Bottleneck cost: max edge weight that would be introduced by choosing c
    
    The bottleneck term penalizes candidates that would create very long edges,
    preventing the tour from having any single dominant long edge. This is a
    fundamentally different objective (minimize max edge vs minimize sum) blended
    with the original sum objective.
    
    Score for candidate c (index i in unvisited):
      cumulative_score(i) + alpha * bottleneck_penalty(i)
    
    where:
      cumulative_score(i) = d(cur,c) + min_{j!=i}[d(c,c_j) + gamma*d(c_j,dest)]
      bottleneck_penalty(i) = max(d(cur,c), min_{j!=i} d(c,c_j), gamma*d(c,dest))
    
    alpha adapts: stronger early (prevent fragmentation), weaker late (destination dominates).
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    if time.time() - _start_time > _LIMIT:
        # Fallback to nearest neighbor for safety
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        c1 = int(unvisited_nodes[0])
        c2 = int(unvisited_nodes[1])
        # Exact calculation for 2 nodes: cur->c1->c2->dest vs cur->c2->c1->dest
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        return c1 if cost1 < cost2 else c2

    # Adaptive parameters
    # gamma: weight for distance to destination in the lookahead.
    # As n_unvisited decreases, we care more about getting close to the destination.
    if n_unvisited > 10:
        gamma = 0.0
    else:
        gamma = (10 - n_unvisited) / 8.0 
        if gamma < 0: gamma = 0
        if gamma > 1: gamma = 1

    # alpha: weight for the bottleneck (minimax) penalty.
    # Stronger when many nodes remain to prevent any single long edge from forming.
    # Weakens as we get closer to the end where the few remaining edges dominate.
    # Scale: starts around 0.3 when n=50, tapers to ~0.05 when n=2.
    alpha = 0.3 * (n_unvisited / 50.0) + 0.02
    
    # Precompute distances
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Submatrix of distances among unvisited nodes
    submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # --- Cumulative Score: 2-step lookahead ---
    # M[i, j] = d(i, j) + gamma * d(j, dest)
    M = submatrix + gamma * dists_to_dest[np.newaxis, :]
    
    # We need min over j != i for each i.
    M_copy = M.copy()
    np.fill_diagonal(M_copy, np.inf)
    
    # inner_mins[i] = min_{j != i} M[i, j]
    inner_mins = np.min(M_copy, axis=1)
    
    # Cumulative score component
    cumulative_scores = dists_from_current + inner_mins
    
    # --- Bottleneck Penalty: Minimax Component ---
    # For each candidate i, the bottleneck is the maximum edge that would be
    # "committed" by choosing i as the next node.
    # Edges involved: d(cur, c_i), the best next edge from c_i to some c_j, and d(c_i, dest) if gamma>0.
    
    # min_{j != i} d(i, j) - the best edge from candidate i to any other unvisited
    sub_for_min = submatrix.copy()
    np.fill_diagonal(sub_for_min, np.inf)
    min_next_edge = np.min(sub_for_min, axis=1)
    
    # Bottleneck for candidate i: max of (d(cur,i), min_next_edge[i], gamma*d(i,dest))
    # Note: d(i,dest) only matters if gamma > 0; when gamma=0, the dest edge isn't in the lookahead
    if gamma > 0:
        bottleneck_terms = np.stack([
            dists_from_current,
            min_next_edge,
            gamma * dists_to_dest
        ], axis=1)
    else:
        bottleneck_terms = np.stack([
            dists_from_current,
            min_next_edge
        ], axis=1)
    
    bottleneck_penalties = np.max(bottleneck_terms, axis=1)
    
    # --- Hybrid Score ---
    # Combine cumulative and bottleneck objectives
    scores = cumulative_scores + alpha * bottleneck_penalties
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
