import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        # Should not happen based on problem description, but safe to return -1 or similar
        # The interface expects an int. If no nodes, maybe return current? 
        # But usually called only if unvisited is not empty.
        return -1 

    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # 1. Get distances from current to all unvisited
    direct_costs = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Get sub-matrix of distances among unvisited nodes
    # This is an NxN matrix where N is len(unvisited_nodes)
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # 3. For each candidate, find the nearest other unvisited node
    # Set diagonal to inf so we don't pick the node itself
    # We need to be careful not to modify the original distance_matrix
    # np.ix_ returns a view or copy? It's a fancy indexing result, usually a copy.
    # To be safe, we can work on a copy or just use a mask.
    # Since N is small (<=50), copying is fine.
    sub_matrix = sub_matrix.copy()
    np.fill_diagonal(sub_matrix, np.inf)
    
    # Min distance to any other unvisited node
    lookahead_costs = np.min(sub_matrix, axis=1)
    
    # 4. Combine costs
    # We use a simple sum. We could weight it, but 1.0 is a good start.
    total_costs = direct_costs + lookahead_costs
    
    # 5. Select the best candidate
    best_local_idx = np.argmin(total_costs)
    
    return int(unvisited_nodes[best_local_idx])
