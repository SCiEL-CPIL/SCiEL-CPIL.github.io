"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for CVRP: Greedy selection minimizing d(current, next) + d(next, depot).
    This balances proximity to the current node with the cost of returning to the depot,
    leading to more compact routes than simple Nearest Neighbor.
    """
    if not unvisited_nodes:
        return depot

    unvisited = np.asarray(unvisited_nodes)
    if len(unvisited) == 0:
        return depot

    # Filter for capacity feasibility
    # Ensure demands is array-like for advanced indexing
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # Calculate heuristic score: distance to candidate + distance from candidate to depot
    # This approximates the cost of extending the route to this node and then closing it.
    dist_cur_to_next = distance_matrix[current_node, feasible_nodes]
    dist_next_to_depot = distance_matrix[feasible_nodes, depot]
    
    scores = dist_cur_to_next + dist_next_to_depot
    
    # Select the node with the minimum score
    min_idx = np.argmin(scores)
    next_node = feasible_nodes[min_idx]
    
    return int(next_node)
# EVOLVE-BLOCK-END
