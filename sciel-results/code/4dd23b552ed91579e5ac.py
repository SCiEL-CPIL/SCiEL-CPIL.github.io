"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 145.0


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard
    if not _time_ok():
        if len(unvisited_nodes) == 0:
            return int(depot)
        unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
        feas_mask = demands[unvisited_arr] <= rest_capacity
        if not np.any(feas_mask):
            return int(depot)
        feas = unvisited_arr[feas_mask]
        return int(feas[np.argmin(distance_matrix[current_node, feas])])

    if len(unvisited_nodes) == 0:
        return int(depot)

    # Convert to numpy array
    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)

    # Filter feasible nodes based on capacity
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)

    feas_nodes = unvisited_arr[feas_mask]
    n_feas = len(feas_nodes)

    if n_feas == 1:
        return int(feas_nodes[0])

    # Distances
    d_cur_dep = distance_matrix[current_node, depot]
    d_dep_feas = distance_matrix[depot, feas_nodes]
    d_cur_feas = distance_matrix[current_node, feas_nodes]

    # Savings heuristic:
    # Saving = d[current, depot] + d[depot, c] - d[current, c]
    # We want to maximize this.
    savings = d_cur_dep + d_dep_feas - d_cur_feas

    # Calculate lookahead connectivity cost
    # For each feasible node c, find the minimum distance to any other feasible node c'
    # This encourages picking nodes that are part of a cluster, reducing travel time
    # between subsequent nodes.
    # If n_feas is small, this is cheap.
    
    # Get the submatrix of distances between feasible nodes
    d_feas_matrix = distance_matrix[np.ix_(feas_nodes, feas_nodes)]
    
    # Set diagonal to inf to exclude self
    np.fill_diagonal(d_feas_matrix, np.inf)
    
    # Min distance from each feasible node to any other feasible node
    if n_feas > 1:
        min_connectivity = np.min(d_feas_matrix, axis=1)
    else:
        min_connectivity = np.zeros(n_feas)

    # Compute tightness: how close we are to filling the vehicle
    # Using average demand of feasible nodes for a smoother tightness measure
    feas_demands = demands[feas_nodes]
    avg_demand = np.mean(feas_demands) if len(feas_demands) > 0 else 1
    if avg_demand > 0:
        tightness = 1.0 - (rest_capacity / avg_demand)
        tightness = np.clip(tightness, 0.0, 1.0)
    else:
        tightness = 0.0

    # Weights tuning
    # savings is primary (maximize -> minimize -savings)
    # d_cur_feas is secondary (minimize)
    # d_dep_feas is tertiary, scaled by tightness (minimize)
    # min_connectivity is quaternary (minimize) - encourages clustering
    
    w1 = 0.15
    w2 = 0.4
    w3 = 0.1 # Weight for connectivity lookahead

    scores = -savings + w1 * d_cur_feas + w2 * tightness * d_dep_feas + w3 * min_connectivity

    best_idx = np.argmin(scores)
    best_node = feas_nodes[best_idx]

    return int(best_node)
# EVOLVE-BLOCK-END
