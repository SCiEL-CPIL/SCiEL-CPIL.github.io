import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    M = len(unvisited_nodes)
    
    if M == 0:
        return -1 # Should not happen based on contract, but safe
        
    if M == 1:
        return int(unvisited_nodes[0])
        
    best_node = -1
    best_cost = float('inf')
    
    # Pre-extract distances from current to all unvisited for speed
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    for i, c in enumerate(unvisited_nodes):
        # Cost to go from current to c
        cost_to_c = dists_from_current[i]
        
        # If c is the last node, we must go to destination
        if M == 2:
            # The other node is the only other option
            # We don't need to look ahead to a "next" unvisited because there are none left after c
            # The path will be current -> c -> destination
            # So cost is d(current, c) + d(c, destination)
            total_cost = cost_to_c + distance_matrix[c, destination_node]
        else:
            # Find the nearest neighbor of c among unvisited nodes excluding c
            # unvisited_nodes is an array. We need to mask out c.
            # Since M is small, we can just loop or use numpy.
            
            # Create a mask of all unvisited except c
            # Actually, we can just find the min distance from c to all unvisited, 
            # but we must ensure we don't pick c itself (distance 0).
            # d(c, c) is 0. So min will be 0 if c is in the list.
            # We want the min distance to *other* nodes.
            
            dists_from_c = distance_matrix[c, unvisited_nodes]
            # Set distance to self to infinity to exclude it
            dists_from_c[i] = float('inf')
            
            min_dist_from_c = np.min(dists_from_c)
            
            total_cost = cost_to_c + min_dist_from_c
            
        if total_cost < best_cost:
            best_cost = total_cost
            best_node = c
            
    return int(best_node)
