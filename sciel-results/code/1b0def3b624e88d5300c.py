"""New program: 2-Step Lookahead Heuristic for CVRP."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    unvisited = np.asarray(unvisited_nodes, dtype=int)
    if unvisited.size == 0:
        return int(depot)

    # Filter to capacity-feasible customers.
    feas = unvisited[demands[unvisited] <= rest_capacity]
    if feas.size == 0:
        return int(depot)

    best_score = float('inf')
    best_node = int(feas[0])
    
    # Get distances from current to all feasible
    d_cur = distance_matrix[current_node, feas]
    
    # Pre-compute distances between feasible nodes? 
    # For N=50, we can just index distance_matrix directly.
    
    for i, c in enumerate(feas):
        # Cost to reach c
        cost_1 = d_cur[i]
        
        # Determine the cost of the next step
        # If this is the only feasible node, we must return to depot.
        if feas.size == 1:
            cost_2 = distance_matrix[c, depot]
        else:
            # Find the nearest neighbor among the *remaining* feasible nodes
            # We need to exclude c from the candidates for the next step
            remaining_mask = np.ones(feas.size, dtype=bool)
            remaining_mask[i] = False
            remaining_feas = feas[remaining_mask]
            
            if remaining_feas.size > 0:
                d_next = distance_matrix[c, remaining_feas]
                cost_2 = np.min(d_next)
            else:
                # Should not happen if size > 1 and we removed one, 
                # but just in case logic error
                cost_2 = distance_matrix[c, depot]
                
        total_score = cost_1 + cost_2
        
        if total_score < best_score:
            best_score = total_score
            best_node = int(c)
            
    return best_node
# EVOLVE-BLOCK-END
