"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


import itertools

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a hybrid heuristic:
    1. Exact brute-force search for the last 8 nodes to optimize the tour tail.
    2. Greedy 1-step lookahead for the remaining nodes to avoid isolated jumps.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    # Exact search for small remaining sets (<= 8 nodes)
    if n_unvisited <= 8:
        best_cost = float('inf')
        best_first = -1
        
        nodes_list = list(unvisited_nodes)
        
        for perm in itertools.permutations(nodes_list):
            # Calculate cost: current -> perm[0] -> ... -> perm[-1] -> destination
            cost = distance_matrix[current_node, perm[0]]
            for i in range(len(perm) - 1):
                cost += distance_matrix[perm[i], perm[i+1]]
            cost += distance_matrix[perm[-1], destination_node]
            
            if cost < best_cost:
                best_cost = cost
                best_first = perm[0]
                
        return int(best_first)

    # Greedy with 1-step lookahead for larger sets
    # Minimize dist(current, u) + min_{v in unvisited \ {u}} dist(u, v)
    
    # Distance from current to all unvisited
    dist_current_u = distance_matrix[current_node, unvisited_nodes]
    
    # Distance matrix among unvisited nodes
    D = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each node u, find the minimum distance to any other unvisited node v (v != u)
    # Set diagonal to inf to exclude self
    D_copy = D.copy()
    np.fill_diagonal(D_copy, np.inf)
    
    min_next_dists = np.min(D_copy, axis=1)
    
    scores = dist_current_u + min_next_dists
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
