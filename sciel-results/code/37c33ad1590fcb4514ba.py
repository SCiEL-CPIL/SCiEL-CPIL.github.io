import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for CVRP: Greedy Savings.
    Picks the next node j that maximizes the savings:
    S(j) = d(current, depot) + d(depot, j) - d(current, j)
    If the maximum saving is negative, it is more efficient to return to the depot
    and start a new route for the remaining nodes.
    """
    if len(unvisited_nodes) == 0:
        return depot
    
    # Identify feasible nodes: those whose demand fits in remaining capacity
    # unvisited_nodes is a list or array of node indices
    feasible_mask = np.array([demands[node] <= rest_capacity for node in unvisited_nodes])
    
    if not np.any(feasible_mask):
        return depot
        
    feasible_nodes = np.array(unvisited_nodes)[feasible_mask]
    
    # Calculate distances from current node to feasible nodes
    dist_current_to_j = distance_matrix[current_node, feasible_nodes]
    
    # Calculate distances from depot to feasible nodes
    dist_depot_to_j = distance_matrix[depot, feasible_nodes]
    
    # Distance from current node to depot
    dist_current_to_depot = distance_matrix[current_node, depot]
    
    # Calculate savings for each feasible node
    # Saving = (Cost to go back to depot and then to j) - (Cost to go directly to j)
    # If we go directly to j, we add dist(current, j).
    # If we go back to depot, we add dist(current, depot). Then from depot to j is dist(depot, j).
    # The "saving" of keeping the route open is dist(current, depot) + dist(depot, j) - dist(current, j).
    savings = dist_current_to_depot + dist_depot_to_j - dist_current_to_j
    
    # Find the node with the maximum saving
    max_saving_idx = np.argmax(savings)
    max_saving = savings[max_saving_idx]
    
    # If the best saving is negative, it is cheaper to return to depot
    # because d(current, j) > d(current, depot) + d(depot, j)
    if max_saving < 0:
        return depot
        
    best_node = feasible_nodes[max_saving_idx]
    
    return int(best_node)
