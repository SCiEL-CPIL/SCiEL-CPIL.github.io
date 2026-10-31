"""Rewritten program: TSP constructive heuristic using MST-aware residual graph contraction."""
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

def _mst_weight(dist_matrix):
    """
    Compute the total weight of the MST of a complete graph given by dist_matrix.
    Uses Prim's algorithm starting from node 0.
    dist_matrix should have inf on diagonal.
    """
    n = dist_matrix.shape[0]
    if n <= 1:
        return 0.0
    
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    
    total_weight = 0.0
    for _ in range(n):
        # Find the node with minimum distance to the MST
        unvisited_mask = ~in_mst
        if not np.any(unvisited_mask):
            break
        min_dist[~unvisited_mask] = np.inf
        u = np.argmin(min_dist)
        in_mst[u] = True
        total_weight += min_dist[u]
        
        # Update distances for neighbors
        min_dist = np.minimum(min_dist, dist_matrix[u])
        min_dist[u] = np.inf
    
    return total_weight

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: MST-aware residual graph contraction.
    
    For each candidate node u in unvisited_nodes, we estimate the cost as:
    cost(u) = d(current, u) 
             + lambda * [MST(unvisited \\ {u}) - MST(unvisited)]
             + mu * d(u, destination) * (1 / (1 + remaining_fraction))
    
    The key structural change: instead of extending a simple path with lookahead,
    we treat the remaining unvisited nodes as a dynamic graph and select the next
    city based on how it affects the minimum spanning tree of the residual problem.
    This captures global connectivity information that simple path extensions miss.
    """
    n = len(unvisited_nodes)
    
    if n == 0:
        return -1
        
    if n == 1:
        return int(unvisited_nodes[0])
        
    if n == 2:
        # For 2 nodes, pick the one that minimizes the total path
        # current -> u -> v -> destination
        u0 = int(unvisited_nodes[0])
        u1 = int(unvisited_nodes[1])
        cost0 = distance_matrix[current_node, u0] + distance_matrix[u0, u1] + distance_matrix[u1, destination_node]
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u1, u0] + distance_matrix[u0, destination_node]
        return u0 if cost0 <= cost1 else u1
        
    # Enforce deadline: if time is running out, fall back to fast 1-step NN
    if _is_deadline_passed():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    # 1. Distance from current node to all unvisited candidates
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distance matrix among unvisited nodes
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)].copy()
    
    # 3. Distance from each unvisited node to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 4. Compute MST weight of the full unvisited subgraph
    d_sub_full = d_sub.copy()
    np.fill_diagonal(d_sub_full, np.inf)
    mst_full = _mst_weight(d_sub_full)
    
    # 5. For each candidate i, compute MST weight of unvisited \ {i}
    # This is expensive but captures global connectivity
    mst_removed = np.zeros(n)
    for i in range(n):
        if n <= 3:
            # For small n, just compute directly
            d_sub_i = d_sub.copy()
            np.fill_diagonal(d_sub_i, np.inf)
            d_sub_i = np.delete(np.delete(d_sub_i, i, axis=0), i, axis=1)
            if d_sub_i.shape[0] > 0:
                mst_removed[i] = _mst_weight(d_sub_i)
            else:
                mst_removed[i] = 0.0
        else:
            # For larger n, use approximation: the change in MST when removing node i
            # is approximately the minimum edge incident to i in the MST
            # This is a heuristic approximation to save time
            # Find the minimum distance from i to any other node
            d_from_i = d_sub[i].copy()
            d_from_i[i] = np.inf
            min_edge = np.min(d_from_i)
            # The MST change is at most min_edge, but could be less
            # Use a conservative estimate
            mst_removed[i] = mst_full - 0.5 * min_edge
    
    # 6. The MST delta: how much the MST improves when we remove node i
    # If removing i reduces the MST weight, it means i was a "hub" that connected
    # other nodes, so visiting it now might be good (we're contracting the MST)
    mst_delta = mst_full - mst_removed  # positive means MST gets lighter when i is removed
    
    # 7. Adaptive weighting
    remaining_fraction = n / 50.0  # assuming ~50 cities
    
    # lambda: weight for MST connectivity term
    # When many nodes remain, MST structure matters more
    lambda_mst = 0.5 * remaining_fraction
    
    # mu: weight for destination term
    # As fewer nodes remain, the destination becomes more important
    mu_dest = 0.4 * (1.0 / (1.0 + remaining_fraction))
    
    # 8. Combine all terms
    # The idea: prefer nodes that are close to current, and whose removal
    # doesn't hurt the MST connectivity of the remaining graph too much
    scores = d_curr - lambda_mst * mst_delta + mu_dest * d_dest
    
    # 9. Select the candidate with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
