import numpy as np
from itertools import permutations

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    # If very few nodes remain, solve exactly for the optimal tail
    # to minimize the final connection cost.
    if n_unvisited <= 5:
        # Get indices of unvisited nodes
        # We need to evaluate permutations of these nodes
        # Cost = d(current, p0) + d(p0, p1) + ... + d(p_last, destination)
        
        best_cost = np.inf
        best_first = -1
        
        # For n=5, 120 permutations is trivial.
        for p in permutations(unvisited_nodes):
            cost = distance_matrix[current_node, p[0]]
            for i in range(len(p) - 1):
                cost += distance_matrix[p[i], p[i+1]]
            # Close the tour to destination
            cost += distance_matrix[p[-1], destination_node]
            
            if cost < best_cost:
                best_cost = cost
                best_first = p[0]
        
        return int(best_first)

    # For larger sets, use 2-step lookahead
    # Score(j) = d(current, j) + min_k d(j, k)
    
    # Vectorized 2-step lookahead
    dists_current = distance_matrix[current_node, unvisited_nodes]
    
    sub_matrix = distance_matrix[unvisited_nodes, unvisited_nodes]
    np.fill_diagonal(sub_matrix, np.inf)
    min_next_dists = np.min(sub_matrix, axis=1)
    
    scores = dists_current + min_next_dists
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
