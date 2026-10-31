"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a 2-step lookahead heuristic.
    
    This approach evaluates each candidate node by estimating the cost of moving
    from the current node to the candidate, and then to the nearest remaining
    unvisited node (or the destination if it's the last one). This helps avoid
    greedy traps where the immediate nearest neighbor leads to a suboptimal tour.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        n1, n2 = int(unvisited_nodes[0]), int(unvisited_nodes[1])
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return n1 if cost1 <= cost2 else n2

    best_node = -1
    min_cost = np.inf
    
    # Iterate through all unvisited nodes
    for u in unvisited_nodes:
        u = int(u)
        
        # Cost to reach u from current
        cost_to_u = distance_matrix[current_node, u]
        
        if cost_to_u >= min_cost:
            # Optimization: if we already have a better cost, skip lookahead
            # But we need to be careful if this is the last node, etc.
            # Actually, let's just compute the full estimate for simplicity and correctness first.
            pass
        
        # Determine the cost of the next step from u
        if n_unvisited == 1:
            # This case is handled above, but for safety in loop logic if n_unvisited changed? 
            # No, n_unvisited is fixed in this call.
            next_cost = distance_matrix[u, destination_node]
        else:
            # Find the nearest neighbor of u among the other unvisited nodes
            # We need to exclude u from the list
            # unvisited_nodes is an array.
            # Create a mask or list comprehension
            # For efficiency with numpy, we can do:
            # others = unvisited_nodes[unvisited_nodes != u]
            # But unvisited_nodes might have duplicates? No, it's a set of unvisited nodes.
            # Let's use a boolean mask for speed.
            mask = unvisited_nodes != u
            others = unvisited_nodes[mask]
            
            if len(others) > 0:
                # Find min distance from u to any node in others
                dists = distance_matrix[u, others]
                min_next_dist = np.min(dists)
            else:
                # Should not happen if n_unvisited > 1
                min_next_dist = distance_matrix[u, destination_node]
            
            next_cost = min_next_dist
        
        total_cost = cost_to_u + next_cost
        
        if total_cost < min_cost:
            min_cost = total_cost
            best_node = u
            
    return best_node
# EVOLVE-BLOCK-END
