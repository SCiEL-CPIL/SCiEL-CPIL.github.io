"""Rewritten program: TSP constructive heuristic with 2-step lookahead and exact endgame."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

_start_time = time.time()

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy:
    1. If few nodes remain (k <= 8), use exact permutation search to find the optimal 
       completion of the tour from the current node through remaining nodes back to destination.
    2. Otherwise, use a 2-step lookahead heuristic: minimize d(current, c) + min_{c' != c} d(c, c').
       This helps avoid choosing nodes that are close to current but far from the rest of the unvisited set.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Ensure indices are integers for numpy indexing
    current_node = int(current_node)
    destination_node = int(destination_node)
    
    # Endgame: Exact solution for small number of remaining nodes
    # 8! = 40,320 permutations, which is computationally feasible in Python
    if n_unvisited <= 8:
        nodes = unvisited_nodes.astype(int)
        best_cost = float('inf')
        best_first_node = -1
        
        # Iterate through all permutations of the remaining nodes
        for perm in itertools.permutations(nodes):
            # Calculate path cost: current -> perm[0] -> ... -> perm[-1] -> destination
            # Start with distance from current to first node in permutation
            cost = distance_matrix[current_node, perm[0]]
            
            # Add distances between consecutive nodes in the permutation
            for i in range(len(perm) - 1):
                cost += distance_matrix[perm[i], perm[i+1]]
                
            # Add distance from last node to destination
            cost += distance_matrix[perm[-1], destination_node]
            
            if cost < best_cost:
                best_cost = cost
                best_first_node = perm[0]
                
        return int(best_first_node)
    
    # Main phase: 2-step Lookahead
    # 1. Distance from current node to all unvisited nodes
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # 2. For each unvisited node, find the distance to its nearest other unvisited node
    # Extract the submatrix of distances between unvisited nodes
    # Using np.ix_ for advanced indexing
    unvisited_idx = unvisited_nodes.astype(int)
    sub_matrix = distance_matrix[np.ix_(unvisited_idx, unvisited_idx)]
    
    # We need the minimum distance to *another* node, so we set the diagonal to infinity
    # to exclude the self-distance (which is 0) from the minimum calculation
    sub_matrix = sub_matrix.copy()
    np.fill_diagonal(sub_matrix, np.inf)
    
    # Minimum distance to the next best node for each candidate
    d_next = np.min(sub_matrix, axis=1)
    
    # Total score: cost to reach candidate + cost to reach the next best node from there
    scores = d_curr + d_next
    
    # Select the candidate with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
