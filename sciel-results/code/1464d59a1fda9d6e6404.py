import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if not unvisited_nodes:
        return 0
        
    # Filter unvisited nodes that fit in remaining capacity
    feasible_nodes = []
    for u in unvisited_nodes:
        if demands[u] <= rest_capacity:
            feasible_nodes.append(u)
            
    if not feasible_nodes:
        return 0
        
    feasible_nodes = np.array(feasible_nodes)
    
    # Heuristic: Minimize the incremental cost of going to u and then returning to depot.
    # Cost = dist(current, u) + dist(u, depot)
    # This balances proximity to current node with proximity to depot (to avoid long tails).
    
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    scores = dist_from_current + dist_to_depot
    
    best_idx = np.argmin(scores)
    best_node = feasible_nodes[best_idx]
    
    return int(best_node)
