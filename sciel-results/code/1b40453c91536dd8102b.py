"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Get feasible nodes based on remaining capacity
    # demands is an array/list where demands[i] is the demand of node i
    # unvisited_nodes is a list/array of node indices
    
    if len(unvisited_nodes) == 0:
        return depot

    # Filter nodes that fit in the remaining capacity
    feasible_nodes = []
    for node in unvisited_nodes:
        if demands[node] <= rest_capacity:
            feasible_nodes.append(node)
            
    if not feasible_nodes:
        return depot

    # Convert to numpy array for vectorized operations if needed, 
    # but loop is fine for N=50
    # dist_current_depot is the cost to close the route now
    dist_current_depot = distance_matrix[current_node, depot]
    
    best_node = None
    best_score = float('inf')
    
    for node in feasible_nodes:
        dist_current_node = distance_matrix[current_node, node]
        dist_node_depot = distance_matrix[node, depot]
        
        # Heuristic: Only consider nodes that are "closer" than going back to depot
        # This prevents long detours away from the depot
        if dist_current_node < dist_current_depot:
            # Score: Total distance added to the route if we visit this node and then return to depot
            # This encourages compact routes
            score = dist_current_node + dist_node_depot
            if score < best_score:
                best_score = score
                best_node = node

    # If no node passed the filter, or best_node is still None, return depot
    if best_node is None:
        return depot
        
    return int(best_node)
# EVOLVE-BLOCK-END
