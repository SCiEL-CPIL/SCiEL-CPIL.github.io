import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if len(unvisited_nodes) == 0:
        return depot

    unvisited = np.array(unvisited_nodes)
    unvisited_demands = demands[unvisited]
    
    # Feasible nodes: demand <= rest_capacity
    feasible_mask = unvisited_demands <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot

    feasible_indices = np.where(feasible_mask)[0]
    feasible_nodes = unvisited[feasible_indices]
    M = len(feasible_nodes)

    if M == 1:
        return int(feasible_nodes[0])

    # Distance from current to all feasible nodes
    dist_current_to_feasible = distance_matrix[current_node, feasible_nodes]
    
    # Distance from all feasible nodes to depot
    dist_feasible_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Distance matrix between feasible nodes
    # shape (M, M)
    dist_between_feasible = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    # We want to estimate the cost after visiting node j.
    # Heuristic: Visit the best next node k, then return to depot.
    # Cost(j) = dist(current, j) + min_{k != j} (dist(j, k) + dist(k, depot))
    
    # Create a copy to modify diagonal
    dist_between_feasible_mod = dist_between_feasible.copy()
    
    # Set diagonal to infinity to prevent selecting self
    np.fill_diagonal(dist_between_feasible_mod, np.inf)
    
    # Add distance from k to depot for each k
    # dist_between_feasible_mod[j, k] + dist_feasible_to_depot[k]
    # We want min over k for each j
    future_costs = np.min(dist_between_feasible_mod + dist_feasible_to_depot[None, :], axis=1)
    
    # Total score for each feasible node j
    total_scores = dist_current_to_feasible + future_costs
    
    # Pick the node with the lowest total score
    best_idx = np.argmin(total_scores)
    
    return int(feasible_nodes[best_idx])
