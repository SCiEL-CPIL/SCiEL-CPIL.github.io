import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # 1. Compute MST of the current unvisited nodes
    # Extract submatrix
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Compute MST
    try:
        mst = minimum_spanning_tree(sub_matrix)
        # Get edge weights
        weights = mst.data
        if len(weights) == 0:
            w_min = 0.0
        else:
            w_min = np.min(weights)
    except Exception:
        w_min = 0.0
    
    # 2. Estimate residual cost using w_min
    # The residual tour must visit n_unvisited - 1 nodes.
    # A lower bound on the length of a path visiting k nodes is (k-1) * min_edge_weight?
    # Actually, the path length is at least (k-1) * min_edge_weight in the complete graph?
    # The MST weight is a better lower bound for the *subtree*, but for a *path*,
    # the minimum edge weight is a very loose lower bound.
    # However, the directive asks to use it.
    
    # Let's combine immediate distance and a scaled residual estimate.
    # Score(u) = dist(c, u) + dist(u, dest) + w_min * (n_unvisited - 1)
    
    best_score = float('inf')
    best_node = -1
    
    # Precompute distances from current and destination to all unvisited
    dist_from_curr = distance_matrix[current_node, unvisited_nodes]
    dist_from_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # The residual term is constant for all candidates if we use global w_min
    residual_est = w_min * (n_unvisited - 1)
    
    for i in range(n_unvisited):
        candidate = unvisited_nodes[i]
        immediate = dist_from_curr[i]
        exit_cost = dist_from_dest[i]
        
        total_score = immediate + exit_cost + residual_est
        
        if total_score < best_score:
            best_score = total_score
            best_node = candidate
            
    return int(best_node)
