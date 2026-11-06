"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: 150s limit, stop at 145s
    if time.time() - _start_time > 145:
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    n_unvisited = len(unvisited_arr)
    
    # Filter feasible nodes by capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return int(depot)
    
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # Pre-compute distances from current node to feasible nodes
    d_current_to_feasible = distance_matrix[current_node, feasible_nodes]
    
    # Pre-compute distances from feasible nodes to depot
    d_feasible_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Pre-compute demands of feasible nodes
    d_feasible_demands = demands[feasible_nodes]
    
    # Heuristic: Weighted Nearest Neighbor with Capacity Penalty
    # Score(i) = d(current, i) + penalty * d(i, depot) / rest_capacity
    # This encourages visiting nearby nodes, but penalizes nodes far from the depot
    # when capacity is running low, to avoid long return trips.
    
    # Calculate base distance from current to feasible nodes
    d_c_i = d_current_to_feasible
    
    # Calculate penalty term
    # As rest_capacity decreases, the penalty for being far from depot increases.
    # We use a coefficient to balance the distance scale.
    # Distances are typically 0-100.
    # If rest_capacity is 50, penalty is small. If 5, penalty is large.
    penalty_weight = 2.0
    penalty_term = penalty_weight * (d_feasible_to_depot / rest_capacity)
    
    scores = d_c_i + penalty_term
    
    # Select the best node
    best_idx = int(np.argmin(scores))
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
