import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if not unvisited_nodes:
        return depot

    # Ensure unvisited_nodes is a numpy array for efficient indexing
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    
    # Ensure demands is a numpy array for efficient indexing
    demands_arr = np.array(demands, dtype=int)
    
    # Filter by capacity
    feasible_mask = demands_arr[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # Vectorized distance calculation
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Score: Minimize the sum of distance to node and distance from node to depot
    # This is equivalent to maximizing the savings of extending the route
    scores = dist_from_current + dist_to_depot
    
    min_idx = np.argmin(scores)
    
    return int(feasible_nodes[min_idx])
