import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if not unvisited_nodes:
        return depot

    # Convert unvisited nodes to numpy array for efficient indexing
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    
    # Filter for nodes that fit in the remaining capacity
    # demands is assumed to be an array/list indexed by node id
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot
        
    feasible_nodes = unvisited_arr[feasible_mask]
    
    # Calculate distances from current node to all feasible nodes
    dist_current = distance_matrix[current_node, feasible_nodes]
    
    # Calculate distances from all feasible nodes to the depot
    dist_depot = distance_matrix[feasible_nodes, depot]
    
    # Heuristic: Minimize the sum of the distance to the node and the distance from the node to the depot.
    # This encourages the vehicle to stay in a compact region near the depot, 
    # reducing the total route length compared to pure nearest neighbor.
    scores = dist_current + dist_depot
    
    # Select the node with the minimum score
    best_idx = np.argmin(scores)
    
    return int(feasible_nodes[best_idx])
