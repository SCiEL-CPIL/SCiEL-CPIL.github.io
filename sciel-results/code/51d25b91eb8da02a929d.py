import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node to visit in a capacitated vehicle routing problem.
    Uses a heuristic that minimizes the sum of the distance from the current node
    to the candidate node and the distance from the candidate node to the depot.
    This encourages routes that are compact around the depot.
    """
    # Handle case where no nodes are left
    if len(unvisited_nodes) == 0:
        return int(depot)
        
    # Ensure inputs are numpy arrays for efficient vectorized operations
    # In case they are passed as lists
    unvisited_arr = np.array(unvisited_nodes)
    demands_arr = np.array(demands)
    dm = np.array(distance_matrix)
    
    # Get demands for the unvisited nodes
    node_demands = demands_arr[unvisited_arr]
    
    # Filter for feasible nodes (demand <= remaining capacity)
    feasible_mask = node_demands <= rest_capacity
    
    # If no feasible nodes, return to depot to close the route
    if not np.any(feasible_mask):
        return int(depot)
        
    # Get the indices of the feasible nodes
    feasible_nodes = unvisited_arr[feasible_mask]
    
    # Calculate the heuristic score for each feasible node:
    # Score = distance(current, node) + distance(node, depot)
    # This accounts for the cost to reach the node and the cost to eventually return
    # to the depot, promoting shorter, more compact routes.
    
    # Vectorized distance calculations
    d_from_current = dm[current_node, feasible_nodes]
    d_to_depot = dm[feasible_nodes, depot]
    
    scores = d_from_current + d_to_depot
    
    # Find the node with the minimum score
    min_idx = np.argmin(scores)
    
    # Return the node ID as an integer
    return int(feasible_nodes[min_idx])
