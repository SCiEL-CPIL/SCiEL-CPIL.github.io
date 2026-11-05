import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node to visit in a CVRP route.
    
    Args:
        current_node: The node we are currently at.
        depot: The index of the depot (usually 0).
        unvisited_nodes: An iterable of unvisited customer indices.
        rest_capacity: The remaining capacity of the vehicle.
        demands: An array/list of demands for each node.
        distance_matrix: A 2D array of distances between nodes.
        
    Returns:
        The index of the next node to visit, or the depot index to close the route.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Convert to numpy arrays for efficient indexing and operations
    # Handle case where inputs might be lists
    unvisited_arr = np.array(unvisited_nodes)
    
    # Filter nodes that fit in the remaining capacity
    # demands might be a list or array
    node_demands = np.array(demands)[unvisited_arr]
    feasible_mask = node_demands <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot

    feasible_nodes = unvisited_arr[feasible_mask]
    
    # Calculate the cost for each feasible node.
    # Heuristic: Minimize d(current, next) + d(next, depot)
    # This encourages picking nodes that are close to the current position
    # and also close to the depot, reducing the total route length.
    
    d_current_to_next = distance_matrix[current_node, feasible_nodes]
    d_next_to_depot = distance_matrix[feasible_nodes, depot]
    
    scores = d_current_to_next + d_next_to_depot
    
    # Find the node with the minimum score
    min_idx = np.argmin(scores)
    next_node = feasible_nodes[min_idx]
    
    return int(next_node)
