"""CVRP heuristic: Savings with 1-Step Look-Ahead, Demand Efficiency, and Capacity-Aware Closing."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = None

# Hyperparameters
W_LOOKAHEAD = 0.6
W_DEMAND = 0.4
CLOSING_THRESHOLD = 0.0


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a savings heuristic enhanced with look-ahead cost
    and demand efficiency.
    
    Returns 0 (depot) if no feasible node can be added, if the score is below
    threshold, or if the time limit is reached.
    """
    global _start_time, _DEADLINE
    
    if _start_time is None:
        _start_time = time.time()
        _DEADLINE = _start_time + _TIMEOUT
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return int(depot)
    
    # Convert unvisited_nodes to numpy array
    if isinstance(unvisited_nodes, (list, tuple)):
        unvisited_arr = np.array(unvisited_nodes, dtype=int)
    else:
        unvisited_arr = np.asarray(unvisited_nodes, dtype=int)
    
    # If no unvisited nodes, return depot
    if len(unvisited_arr) == 0:
        return int(depot)
    
    # Find feasible nodes
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited_arr[feasible_mask]
    n_feasible = len(feasible_nodes)
    
    # If at depot, pick the nearest feasible node to start the route
    if current_node == depot:
        dists = distance_matrix[current_node, feasible_nodes]
        min_idx = np.argmin(dists)
        return int(feasible_nodes[min_idx])
    
    # Compute base distances
    d_current_depot = distance_matrix[current_node, depot]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    d_current_j = distance_matrix[current_node, feasible_nodes]
    
    # 1. Classic Savings
    savings = d_current_depot + d_depot_j - d_current_j
    
    # 2. Look-Ahead Cost: Cost to go from j to the best next node k
    if n_feasible > 1:
        d_j_k = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
        np.fill_diagonal(d_j_k, np.inf)
        min_dist_to_next = np.min(d_j_k, axis=1)
    else:
        min_dist_to_next = d_depot_j.copy()
        
    lookahead_penalty = W_LOOKAHEAD * min_dist_to_next
    
    # 3. Demand Efficiency Bonus
    feasible_demands = demands[feasible_nodes]
    demand_ratio = feasible_demands / max(1, rest_capacity)
    demand_bonus = W_DEMAND * savings * demand_ratio
    
    # Final Score
    scores = savings - lookahead_penalty + demand_bonus
    
    # Decision: If the best score is not significantly positive, close the route
    if np.max(scores) <= CLOSING_THRESHOLD:
        return int(depot)
    
    best_idx = np.argmax(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
