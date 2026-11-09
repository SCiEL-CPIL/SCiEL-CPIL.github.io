import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    best_score = np.inf
    best_node = -1
    
    # Precompute distances from destination to all unvisited
    # This helps in the inner loop
    dist_dest = distance_matrix[destination_node, unvisited_nodes]
    
    for i, c in enumerate(unvisited_nodes):
        # Immediate cost to come to c
        cost_to_c = distance_matrix[current_node, c]
        
        # If we pick c, the remaining nodes are unvisited_nodes without c
        # We need to estimate the cost of the next step from c.
        # The next node n will be in R = unvisited \ {c}.
        # We approximate the future cost by looking for the best n
        # that minimizes d(c, n) + d(n, destination).
        # This is a greedy 2-step lookahead.
        
        if n_unvisited == 2:
            # Only one other node r
            r_idx = 1 - i
            r_node = unvisited_nodes[r_idx]
            # Path: current -> c -> r -> destination
            # Cost: d(cur, c) + d(c, r) + d(r, dest)
            total_cost = cost_to_c + distance_matrix[c, r_node] + distance_matrix[r_node, destination_node]
            if total_cost < best_score:
                best_score = total_cost
                best_node = c
            continue
        
        # General case
        # Get indices of other nodes
        # We can vectorize the calculation of d(c, n) + d(n, dest) for all n != c
        
        # d(c, unvisited) is a vector of size K
        # d(unvisited, dest) is dist_dest (vector of size K)
        # We want min over j != i of (d(c, unvisited[j]) + dist_dest[j])
        
        dist_c_to_unvisited = distance_matrix[c, unvisited_nodes]
        
        # Combine scores for all potential next nodes n
        # score_n = d(c, n) + d(n, dest)
        scores_next = dist_c_to_unvisited + dist_dest
        
        # Mask out the current node c (index i) so we don't pick it as 'next'
        # We set its score to inf
        scores_next[i] = np.inf
        
        min_next_cost = np.min(scores_next)
        
        # Total estimated cost if we pick c
        # It is cost to enter c + cost to exit c to best next node
        # Note: This is a lower bound on the actual path cost from c onwards,
        # but it's a good heuristic for connectivity.
        total_cost = cost_to_c + min_next_cost
        
        if total_cost < best_score:
            best_score = total_cost
            best_node = c
            
    return int(best_node)
