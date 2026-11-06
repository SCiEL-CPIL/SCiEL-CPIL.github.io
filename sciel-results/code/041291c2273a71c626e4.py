# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    if n_unvisited == 2:
        u1 = unvisited_nodes[0]
        u2 = unvisited_nodes[1]
        # Cost of going u1 -> u2 -> dest
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u1, u2] + distance_matrix[u2, destination_node]
        # Cost of going u2 -> u1 -> dest
        cost2 = distance_matrix[current_node, u2] + distance_matrix[u2, u1] + distance_matrix[u1, destination_node]
        
        if cost1 < cost2:
            return int(u1)
        else:
            return int(u2)

    # General case: 2-step lookahead
    # We want to minimize d(current, v) + d(v, nearest_neighbor_of_v_in_unvisited)
    # Note: This is an approximation of the cost of inserting v and its successor.
    
    best_score = float('inf')
    best_node = unvisited_nodes[0]
    
    # Pre-fetch distances from current node to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    for i, v in enumerate(unvisited_nodes):
        d_cur_v = dists_from_current[i]
        
        # Find nearest neighbor to v among unvisited nodes (excluding v itself)
        # We can use the distance matrix row for v
        dists_from_v = distance_matrix[v, unvisited_nodes]
        
        # Set distance to self to inf to ignore
        # Create a copy to modify, or use np.where
        # Since n_unvisited is small (<=48 in this branch), a simple loop or mask is fine.
        # Mask out v's own distance
        dists_from_v_copy = dists_from_v.copy()
        dists_from_v_copy[i] = np.inf
        
        nearest_neighbor_dist = np.min(dists_from_v_copy)
        
        score = d_cur_v + nearest_neighbor_dist
        
        if score < best_score:
            best_score = score
            best_node = v
            
    return int(best_node)
# EVOLVE-BLOCK-END
