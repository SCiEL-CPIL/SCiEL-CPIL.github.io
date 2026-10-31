"""Rewritten program: TSP constructive heuristic using MST-weight + minimax centrality."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline enforcement
_start_time = time.time()
_TOTAL_LIMIT = 150.0  # Evaluator limit
_MARGIN = 5.0         # Safety margin

def _is_deadline_passed():
    """Check if we are close to the time limit."""
    return (time.time() - _start_time) > (_TOTAL_LIMIT - _MARGIN)

def _mst_weight(nodes, dist_matrix):
    """
    Compute the total weight of the MST over the given nodes using Prim's algorithm.
    This captures global connectivity cost of the remaining subgraph.
    """
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    # Extract submatrix
    sub_dist = dist_matrix[np.ix_(nodes, nodes)]
    
    # Prim's algorithm
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    
    mst_weight = 0.0
    
    for _ in range(n):
        # Find the node with minimum distance not yet in MST
        min_dist[in_mst] = np.inf
        u = np.argmin(min_dist)
        in_mst[u] = True
        mst_weight += min_dist[u]
        
        # Update distances
        min_dist = np.minimum(min_dist, sub_dist[u])
    
    return mst_weight

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: MST-weight + minimax centrality penalty.
    
    For each candidate u, compute:
    1. Immediate step cost: d(current, u)
    2. Residual MST weight: MST weight of the remaining unvisited nodes after picking u.
       This captures how well-connected the remaining graph is.
    3. Minimax centrality penalty: max distance from u to any other unvisited node.
       This penalizes choosing nodes that are isolated from the rest.
    
    score(u) = d(current, u) + w_mst * MST(remaining_after_u) + w_centrality * max_d(u, others)
    """
    n = len(unvisited_nodes)
    
    if n == 0:
        return -1
        
    if n == 1:
        return int(unvisited_nodes[0])
        
    # Enforce deadline: if time is running out, fall back to fast 1-step NN
    if _is_deadline_passed():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    # 1. Distance from current node to all unvisited candidates
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Compute MST weight for each candidate's remaining set
    # For each u, the remaining set is unvisited_nodes without u
    # We need to compute MST for n subsets, each of size n-1
    
    # For efficiency, precompute the full submatrix
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # 3. Compute minimax centrality: max distance from each node to any other unvisited node
    # max_d[u] = max_{v != u} d(u, v)
    max_d = np.max(d_sub, axis=1)
    
    # 4. Compute MST weights for each candidate's remaining set
    # This is expensive, so we use a simplified approach:
    # Approximate MST(remaining) using the sum of min distances from each node to the rest
    # This is a lower bound on MST and cheaper to compute
    
    # For each node u, compute the sum of min distances from each other node v to its nearest neighbor
    # Actually, let's use a better approximation: the sum over all remaining nodes of their min distance
    # to any other remaining node, divided by 2 (since each edge is counted twice)
    
    # Precompute min distance from each node to any other node in the full set
    min_d_to_other = np.min(d_sub, axis=1)
    
    # For candidate u, the remaining set is all except u
    # Approximate MST(remaining) ≈ 0.5 * sum_{v != u} min_{w != v, w != u} d(v, w)
    
    # Compute this for all u at once
    # For each v, we need the min distance to any w != v and w != u
    # This is tricky to vectorize fully, so let's do it in a loop for clarity
    # But n is at most 50, so this should be fast enough
    
    mst_approx = np.zeros(n)
    
    for i in range(n):
        # Remaining set is all nodes except i
        remaining_indices = [j for j in range(n) if j != i]
        if len(remaining_indices) <= 1:
            mst_approx[i] = 0.0
        else:
            # Get the sub-submatrix for remaining nodes
            rem_dist = d_sub[np.ix_(remaining_indices, remaining_indices)]
            # Compute min distance from each node to any other in remaining set
            min_d_rem = np.min(rem_dist, axis=1)
            # Approximate MST as half the sum of min distances
            mst_approx[i] = 0.5 * np.sum(min_d_rem)
    
    # 5. Normalize terms
    # Normalize d_curr
    curr_min = np.min(d_curr)
    curr_max = np.max(d_curr)
    if curr_max > curr_min:
        curr_norm = (d_curr - curr_min) / (curr_max - curr_min)
    else:
        curr_norm = np.zeros(n)
        
    # Normalize mst_approx
    mst_min = np.min(mst_approx)
    mst_max = np.max(mst_approx)
    if mst_max > mst_min:
        mst_norm = (mst_approx - mst_min) / (mst_max - mst_min)
    else:
        mst_norm = np.zeros(n)
        
    # Normalize max_d
    maxd_min = np.min(max_d)
    maxd_max = np.max(max_d)
    if maxd_max > maxd_min:
        maxd_norm = (max_d - maxd_min) / (maxd_max - maxd_min)
    else:
        maxd_norm = np.zeros(n)
        
    # 6. Combine scores
    # Weights:
    # - Immediate distance is the most important.
    # - MST term captures global connectivity of remaining graph.
    # - Minimax centrality penalizes isolated nodes.
    
    w_curr = 1.0
    w_mst = 0.3
    w_centrality = 0.1
    
    scores = w_curr * curr_norm + w_mst * mst_norm + w_centrality * maxd_norm
    
    # 7. Select the candidate with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
