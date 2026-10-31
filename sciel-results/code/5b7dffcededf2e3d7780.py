import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return destination_node
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    if n_unvisited == 2:
        c1 = unvisited_nodes[0]
        c2 = unvisited_nodes[1]
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        
        if cost1 <= cost2:
            return int(c1)
        else:
            return int(c2)
            
    if n_unvisited == 3:
        c1, c2, c3 = unvisited_nodes[0], unvisited_nodes[1], unvisited_nodes[2]
        
        # Calculate best cost if we start with c1
        # Options: c1->c2->c3 or c1->c3->c2
        cost_c1_1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, c3] + distance_matrix[c3, destination_node]
        cost_c1_2 = distance_matrix[current_node, c1] + distance_matrix[c1, c3] + distance_matrix[c3, c2] + distance_matrix[c2, destination_node]
        best_start_c1 = min(cost_c1_1, cost_c1_2)
        
        # Calculate best cost if we start with c2
        # Options: c2->c1->c3 or c2->c3->c1
        cost_c2_1 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, c3] + distance_matrix[c3, destination_node]
        cost_c2_2 = distance_matrix[current_node, c2] + distance_matrix[c2, c3] + distance_matrix[c3, c1] + distance_matrix[c1, destination_node]
        best_start_c2 = min(cost_c2_1, cost_c2_2)
        
        # Calculate best cost if we start with c3
        # Options: c3->c1->c2 or c3->c2->c1
        cost_c3_1 = distance_matrix[current_node, c3] + distance_matrix[c3, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        cost_c3_2 = distance_matrix[current_node, c3] + distance_matrix[c3, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        best_start_c3 = min(cost_c3_1, cost_c3_2)
        
        if best_start_c1 <= best_start_c2 and best_start_c1 <= best_start_c3:
            return int(c1)
        elif best_start_c2 <= best_start_c1 and best_start_c2 <= best_start_c3:
            return int(c2)
        else:
            return int(c3)

    # n_unvisited > 3
    best_cost = float('inf')
    best_next = unvisited_nodes[0]
    
    for i in range(n_unvisited):
        c1 = unvisited_nodes[i]
        d_current_to_c1 = distance_matrix[current_node, c1]
        
        if d_current_to_c1 >= best_cost:
            continue
            
        for j in range(n_unvisited):
            if i == j:
                continue
            c2 = unvisited_nodes[j]
            
            d_c1_to_c2 = distance_matrix[c1, c2]
            partial_cost = d_current_to_c1 + d_c1_to_c2
            
            if partial_cost >= best_cost:
                continue
                
            # Get distances from c2 to all unvisited nodes
            # Using .copy() to avoid modifying the original view/array
            dists = distance_matrix[c2, unvisited_nodes].copy()
            
            # Exclude c1 and c2 from consideration for the next step
            dists[i] = np.inf
            dists[j] = np.inf
            
            min_d = np.min(dists)
            
            total_cost = partial_cost + min_d
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_next = c1
                
    return int(best_next)
