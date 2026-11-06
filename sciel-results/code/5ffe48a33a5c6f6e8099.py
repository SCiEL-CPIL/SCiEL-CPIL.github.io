"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a destination-biased 2-step lookahead heuristic.
    
    Score for candidate u:
    cost = dist(current, u) + min_dist(u, unvisited \ {u}) + bias * dist(u, destination)
    
    The bias term encourages the tour to start converging towards the destination 
    earlier, reducing the risk of a long final edge.
    """
    if len(unvisited_nodes) == 0:
        return destination_node
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])

    # Get distances from current_node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from destination_node to all unvisited nodes
    dist_to_dest = distance_matrix[destination_node, unvisited_nodes]
    
    best_score = np.inf
    best_node = unvisited_nodes[0]
    
    # Bias factor: small enough not to dominate the immediate cost, 
    # but large enough to influence the path.
    bias = 0.1
    
    for i, u in enumerate(unvisited_nodes):
        cost_to_u = dist_from_current[i]
        
        # Calculate min distance from u to other unvisited nodes
        # We need min(d(u, v)) for v in unvisited_nodes, v != u
        dists_from_u = distance_matrix[u, unvisited_nodes]
        
        # Mask out u itself
        if len(unvisited_nodes) > 1:
            # Create mask
            mask = np.ones(len(unvisited_nodes), dtype=bool)
            mask[i] = False
            valid_dists = dists_from_u[mask]
            if len(valid_dists) > 0:
                next_cost = np.min(valid_dists)
            else:
                next_cost = 0.0 # Should not happen if len > 1
        else:
            next_cost = 0.0

        # Add destination bias
        dest_cost = dist_to_dest[i]
        
        total_cost = cost_to_u + next_cost + bias * dest_cost
        
        if total_cost < best_score:
            best_score = total_cost
            best_node = u

    return int(best_node)
# EVOLVE-BLOCK-END
