"""Improved TSP constructive heuristic using enhanced minimax strategy."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using an enhanced minimax strategy.
    
    The cost for each candidate node u is defined as the maximum of:
    1. The distance from the current node to u (immediate edge cost).
    2. The maximum distance from u to any other unvisited node (connectivity cost).
    3. The maximum distance from any *remaining* unvisited node to the destination 
       (closing edge risk reduction).
    
    We choose the candidate that minimizes this cost.
    """
    if len(unvisited_nodes) == 0:
        return int(current_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_unvisited = len(unvisited_nodes)
    
    # 1. Distance from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Pairwise distances among unvisited nodes
    submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Max distance from each unvisited node to any other unvisited node
    # Set diagonal to 0 (or -inf) to exclude self
    np.fill_diagonal(submatrix, 0)
    max_dist_to_others = np.max(submatrix, axis=1)
    
    # 3. Distances from all unvisited nodes to destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Calculate the max distance to destination for the remaining set
    # if we pick node i, the remaining set is U \ {i}
    # The max distance in U \ {i} is:
    # - If i is the index of the global max, it's the second max
    # - Otherwise, it's the global max
    
    # Find global max and second max of dist_to_dest
    # np.argsort is stable enough, or we can use argpartition
    # For small N, simple approach:
    max_val = np.max(dist_to_dest)
    max_idx = np.argmax(dist_to_dest)
    
    # To get second max, we can set the max element to -inf and take max again
    # Or use argpartition
    if n_unvisited > 1:
        # Create a copy to avoid modifying original
        dist_to_dest_copy = dist_to_dest.copy()
        dist_to_dest_copy[max_idx] = -np.inf
        second_max_val = np.max(dist_to_dest_copy)
    else:
        second_max_val = 0.0
        
    # Construct the "remaining max" array
    # For each i, remaining_max[i] = max_{j != i} dist_to_dest[j]
    remaining_max = np.full(n_unvisited, max_val)
    remaining_max[max_idx] = second_max_val
    
    # Total cost for each candidate
    # Cost(u) = max( dist_from_current[u], max_dist_to_others[u], remaining_max[u] )
    costs = np.maximum(dist_from_current, max_dist_to_others)
    costs = np.maximum(costs, remaining_max)
    
    # Choose the candidate with the minimum cost
    best_idx = int(np.argmin(costs))
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
