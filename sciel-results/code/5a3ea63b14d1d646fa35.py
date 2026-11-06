"""Rewritten program: Nearest Neighbor with Depot Bias heuristic for node selection."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer (or depot 0 to close) using a Nearest Neighbor heuristic
    with a bias towards customers closer to the depot.
    Score to minimize: distance(current, j) + lambda * distance(j, depot)
    """
    unvisited = np.asarray(unvisited_nodes, dtype=int)
    n_unvisited = len(unvisited)

    # If no unvisited nodes, close the route.
    if n_unvisited == 0:
        return 0

    # Feasibility mask: demand[j] <= rest_capacity
    feas = demands[unvisited] <= rest_capacity
    if not np.any(feas):
        return 0

    # Restrict to feasible candidates
    feas_idx = unvisited[feas]

    # Distances from current_node to each feasible candidate
    dist_cur = distance_matrix[current_node, feas_idx]
    
    # Distances from depot to each feasible candidate
    dist_depot = distance_matrix[depot, feas_idx]

    # Weighted score: primarily nearest neighbor, slightly biased towards depot
    # Lambda = 0.1 means we care 10% as much about being near the depot as being near current
    # This helps compact routes and avoid long detours away from the depot
    lambda_bias = 0.1
    score = dist_cur + lambda_bias * dist_depot

    # Pick the minimum score
    best_local = np.argmin(score)
    return int(feas_idx[best_local])
# EVOLVE-BLOCK-END
