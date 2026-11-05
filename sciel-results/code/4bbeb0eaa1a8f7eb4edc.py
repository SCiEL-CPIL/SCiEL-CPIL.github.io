"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a 2-step lookahead heuristic with capacity awareness.
    Score for node i combines:
    1. Immediate distance cost: dist(current, i)
    2. Lookahead cost: min over feasible j != i of dist(i, j) (penalized)
    3. Closing cost consideration: dist(i, depot) if capacity is tight
    
    Returns 0 (depot) to close a route, or the index of the next customer.
    """
    # Time guard: if we are close to the limit, fall back to fastest heuristic (nearest)
    if _time_remaining() < 0.5:
        # Fastest fallback: Nearest feasible
        if len(unvisited_nodes) == 0:
            return depot
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        if len(feasible) == 1:
            return int(feasible[0])
        dists = distance_matrix[current_node, feasible]
        return int(feasible[np.argmin(dists)])

    if len(unvisited_nodes) == 0:
        return depot

    # Filter feasible nodes based on capacity
    feasible = []
    for n in unvisited_nodes:
        if demands[n] <= rest_capacity:
            feasible.append(n)
    
    if len(feasible) == 0:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    # Convert to numpy array for vectorized distance calculation
    feasible_arr = np.array(feasible)
    n_feasible = len(feasible_arr)
    
    # Calculate distances from current node to all feasible nodes
    d_current_to_feasible = distance_matrix[current_node, feasible_arr]
    d_feasible_to_depot = distance_matrix[feasible_arr, depot]
    
    # 2-step lookahead: for each candidate i, find the minimum distance to another feasible node j
    # This is computed as the row minimum of distance_matrix[feasible_arr, feasible_arr], 
    # but we need to exclude the diagonal (i itself)
    # Create a mask to exclude self-distances
    d_feasible_to_feasible = distance_matrix[np.ix_(feasible_arr, feasible_arr)]
    # Set diagonal to infinity to exclude self
    np.fill_diagonal(d_feasible_to_feasible, np.inf)
    # If there's only one other node, the min is that node's distance
    # If n_feasible == 2, each has exactly one other
    min_next_dist = np.min(d_feasible_to_feasible, axis=1)
    
    # Determine capacity tightness: if rest_capacity is small relative to typical demands,
    # we may need to close the route soon
    # Use a heuristic: if rest_capacity <= max_demand * 1.5, consider closing cost more heavily
    max_demand = np.max(demands[feasible_arr]) if n_feasible > 0 else 0
    capacity_tight = rest_capacity <= max_demand * 1.5
    
    if capacity_tight:
        # Weight closing cost more heavily
        alpha = 0.6  # weight for immediate distance
        beta = 0.2   # weight for lookahead
        gamma = 0.2  # weight for closing
    else:
        # More room to continue, weight lookahead more
        alpha = 0.5
        beta = 0.3
        gamma = 0.2
    
    # Combined score
    scores = alpha * d_current_to_feasible + beta * min_next_dist + gamma * d_feasible_to_depot
    
    # We pick the node with the minimum score.
    min_idx = np.argmin(scores)
    best_node = feasible_arr[min_idx]
    
    # Additional check: if the best score is very high and capacity is tight,
    # consider returning to depot to avoid a very long final leg
    d_current_to_depot = distance_matrix[current_node, depot]
    best_score = scores[min_idx]
    
    # If the best candidate's score is significantly worse than just going to depot,
    # and capacity is tight, close the route
    if capacity_tight and best_score > 1.5 * d_current_to_depot:
        return depot
    
    return int(best_node)

# EVOLVE-BLOCK-END
