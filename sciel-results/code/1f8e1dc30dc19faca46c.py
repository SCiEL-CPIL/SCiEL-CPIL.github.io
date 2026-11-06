import numpy as np
from itertools import permutations

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1 # Should not happen based on contract, but safe
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # For small numbers of unvisited nodes, find the exact best next node
    # by evaluating all permutations of the remaining nodes.
    if n_unvisited <= 6:
        best_cost = float('inf')
        best_first_node = -1
        
        # Generate permutations
        for p in permutations(unvisited_nodes):
            # Path: current -> p[0] -> p[1] -> ... -> p[-1] -> destination
            cost = distance_matrix[current_node, p[0]]
            for i in range(len(p) - 1):
                cost += distance_matrix[p[i], p[i+1]]
            cost += distance_matrix[p[-1], destination_node]
            
            if cost < best_cost:
                best_cost = cost
                best_first_node = p[0]
                
        return int(best_first_node)

    # For larger sets, use a 2-step lookahead heuristic
    U = unvisited_nodes
    
    # Distance from current to all candidates
    d_curr = distance_matrix[current_node, U]
    
    # Distance matrix between all candidates
    # We want min distance from u to any other v in U
    sub = distance_matrix[np.ix_(U, U)].copy()
    np.fill_diagonal(sub, np.inf)
    min_next = np.min(sub, axis=1)
    
    # Score: cost to reach u + cost to reach next best node from u
    scores = d_curr + min_next
    
    best_idx = np.argmin(scores)
    return int(U[best_idx])
