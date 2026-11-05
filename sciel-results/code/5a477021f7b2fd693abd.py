"""CVRP heuristic combining savings, regret, and capacity-aware lookahead."""
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
    
    # Calculate distances
    d_cur_dep = distance_matrix[current_node, depot]
    d_dep_feas = distance_matrix[depot, feas_nodes]
    d_cur_feas = distance_matrix[current_node, feas_nodes]
    
    # Calculate Savings
    # Savings = d[current, depot] + d[depot, c] - d[current, c]
    savings = d_cur_dep + d_dep_feas - d_cur_feas
    
    # 2-Step Lookahead Continuation Cost:
    # For each candidate c, estimate the minimum cost to continue the route
    # after visiting c. This is min(d(c, depot), min_{j != c, j feasible} d(c, j)).
    # This penalizes candidates that are isolated or far from other feasible nodes.
    
    # Get distances from each feasible node to all other feasible nodes
    d_feas_feas = distance_matrix[np.ix_(feas_nodes, feas_nodes)]
    
    # For each node i, find the minimum distance to another feasible node
    # We need to exclude the node itself (diagonal)
    # Set diagonal to infinity
    np.fill_diagonal(d_feas_feas, np.inf)
    
    # Min distance to another feasible node
    min_dist_to_other_feas = np.min(d_feas_feas, axis=1)
    
    # Continuation cost: min of distance to depot or min distance to another feasible node
    cont_cost = np.minimum(d_dep_feas, min_dist_to_other_feas)
    
    # Capacity-aware lookahead:
    # Penalize nodes that have high demand relative to remaining capacity after visiting them
    # This encourages filling routes more completely before closing
    remaining_after_visit = rest_capacity - demands[feas_nodes]
    # Normalize demand by remaining capacity (higher = less remaining capacity)
    cap_pressure = demands[feas_nodes] / (rest_capacity + 1e-9)
    
    # Regret term: penalize edges that are longer than the average
    mean_dist = np.mean(d_cur_feas)
    regret = mean_dist - d_cur_feas  # Favors shorter edges
    
    # Weighted combination:
    # score = savings - W1 * d_cur_feas + W2 * regret - W3 * cont_cost + W4 * (1 - cap_pressure)
    # Higher score is better
    
    W1 = 0.5   # Penalty on direct distance
    W2 = 1.0   # Regret weight
    W3 = 0.3   # Continuation cost weight
    W4 = 0.2   # Capacity pressure weight (encourages filling routes)
    
    scores = savings - W1 * d_cur_feas + W2 * regret - W3 * cont_cost + W4 * (1.0 - cap_pressure)
    
    # Select the node with the highest score
    best_idx = np.argmax(scores)
    best_node = int(feas_nodes[best_idx])
    
    return best_node
# EVOLVE-BLOCK-END
