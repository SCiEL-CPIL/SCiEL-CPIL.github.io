"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 2-step lookahead heuristic.
    
    For each candidate node u in unvisited_nodes, we estimate the cost as:
    dist(current, u) + dist(u, best_next_after_u)
    
    best_next_after_u is the nearest unvisited node to u (excluding current and u).
    If no such node exists, we use dist(u, destination_node).
    
    This helps avoid local minima where the nearest neighbor leads to a costly
    jump away from the main cluster of remaining cities.
    """
    if len(unvisited_nodes) == 0:
        return destination_node
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])

    # Get distances from current_node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    best_score = np.inf
    best_node = unvisited_nodes[0]
    
    # Iterate through each unvisited node to evaluate the 2-step cost
    for i, u in enumerate(unvisited_nodes):
        # Cost to reach u
        cost_to_u = dist_from_current[i]
        
        # If the cost to reach u is already worse than the best found so far,
        # and we know the minimum possible next step distance is >= 0, 
        # we could prune, but for simplicity and robustness with N=50, 
        # we compute the next step.
        
        # Find the nearest unvisited node to u, excluding u itself.
        # Note: We should ideally exclude current_node as well if it's not in unvisited_nodes,
        # but typically unvisited_nodes does not contain current_node.
        # We must exclude u from the candidates for the "next" step.
        
        # Create a mask to exclude u from unvisited_nodes
        # Since unvisited_nodes is a 1D array of indices, we can filter it.
        # However, for efficiency, we can just look at the whole distance row for u
        # and pick the min among unvisited nodes, ensuring we don't pick u.
        
        # Vectorized approach for finding best next from u:
        # dists_from_u = distance_matrix[u, unvisited_nodes]
        # We need min(dists_from_u) where the corresponding node is not u.
        
        dists_from_u = distance_matrix[u, unvisited_nodes]
        
        # Mask out the index i (which corresponds to u)
        # We want the minimum of dists_from_u excluding the element at index i.
        if len(unvisited_nodes) > 1:
            # Create a copy or use np.where
            # np.delete is a bit slow, let's use a mask
            mask = np.ones(len(unvisited_nodes), dtype=bool)
            mask[i] = False
            
            valid_dists = dists_from_u[mask]
            
            if len(valid_dists) > 0:
                next_cost = np.min(valid_dists)
            else:
                # No other unvisited nodes, go to destination
                next_cost = distance_matrix[u, destination_node]
        else:
            # Only one unvisited node (u), so next must be destination
            next_cost = distance_matrix[u, destination_node]

        total_cost = cost_to_u + next_cost
        
        if total_cost < best_score:
            best_score = total_cost
            best_node = u

    return int(best_node)
# EVOLVE-BLOCK-END
