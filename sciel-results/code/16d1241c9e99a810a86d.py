"""Rewritten program: Nearest Neighbor heuristic with capacity-aware closing."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer (or depot 0 to close) using a Nearest Neighbor heuristic.
    If the nearest feasible customer is further than the depot, close the route.
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
    
    # Distance from current_node to depot
    dist_depot = distance_matrix[current_node, depot]

    # Find the nearest feasible candidate
    min_dist_val = np.min(dist_cur)
    min_idx_local = np.argmin(dist_cur)
    
    # Heuristic: If the nearest customer is further than the depot, 
    # it is likely more efficient to return to depot now.
    if min_dist_val > dist_depot:
        return 0

    # Otherwise, go to the nearest feasible customer
    return int(feas_idx[min_idx_local])
# EVOLVE-BLOCK-END
