"""Improved CVRP heuristic with 2-lookahead minimum continuation cost."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 148.0  # Safety margin before 150s limit

def _time_ok():
    return (time.time() - _T0) < _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # If time is running out, return depot to close route safely
    if not _time_ok():
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    # Convert to numpy arrays for efficiency
    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    
    # Filter feasible nodes based on capacity
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)
    
    feas_nodes = unvisited_arr[feas_mask]
    n_nodes = len(feas_nodes)
    
    if n_nodes == 1:
        return int(feas_nodes[0])
    
    # 2-lookahead minimum continuation cost:
    # For each candidate c, estimate the incremental cost as:
    #   cost(c) = d(current, c) + min(d(c, next_best), d(c, depot))
    # where next_best is the nearest other feasible node (or depot if none).
    # This is a different structural approach: instead of savings-based clustering,
    # we directly minimize the 2-step path cost, which naturally avoids
    # picking nodes that lead to expensive continuations.
    
    # Distance from current to each feasible node
    d_cur_feas = distance_matrix[current_node, feas_nodes]
    
    # Distance from depot to each feasible node
    d_dep_feas = distance_matrix[depot, feas_nodes]
    
    # For each candidate i, compute the minimum distance from feas_nodes[i] to any
    # other feasible node j (j != i) or to the depot.
    # This is the "continuation cost" after visiting candidate i.
    
    # Build distance matrix among feasible nodes
    feas_dist = distance_matrix[np.ix_(feas_nodes, feas_nodes)]
    
    # For each row i, find the min distance to any other node (excluding self)
    # and also consider going back to depot
    n = len(feas_nodes)
    
    # Set diagonal to inf so we don't pick self
    np.fill_diagonal(feas_dist, np.inf)
    
    # Min distance from each feasible node to any other feasible node
    min_to_other = np.min(feas_dist, axis=1) if n > 1 else np.array([np.inf])
    
    # Continuation cost: min(distance to next feasible, distance to depot)
    cont_cost = np.minimum(min_to_other, d_dep_feas)
    
    # Total estimated incremental cost for each candidate
    total_cost = d_cur_feas + cont_cost
    
    # Select the node with the lowest total continuation cost
    best_idx = np.argmin(total_cost)
    best_node = int(feas_nodes[best_idx])
    
    return best_node
# EVOLVE-BLOCK-END
