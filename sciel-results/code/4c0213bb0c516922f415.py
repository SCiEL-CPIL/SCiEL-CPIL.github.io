import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if len(unvisited_nodes) == 0:
        return depot
    
    # Filter to feasible nodes (demand fits in remaining capacity)
    feasible = []
    for j in unvisited_nodes:
        if demands[j] <= rest_capacity:
            feasible.append(j)
    
    if len(feasible) == 0:
        return depot
    
    feasible = np.array(feasible)
    
    # Compute score: dist(current, j) + dist(j, depot)
    # This approximates the cost of visiting j and returning to depot
    scores = distance_matrix[current_node, feasible] + distance_matrix[feasible, depot]
    
    # Pick the minimum score
    best_idx = np.argmin(scores)
    return int(feasible[best_idx])
