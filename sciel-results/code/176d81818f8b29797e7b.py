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
    Selects the next node using a 2-step lookahead heuristic with savings-based scoring.
    
    The structural change here is a 2-step lookahead: for each feasible node i, we estimate
    the cost of going current -> i -> (best next step j OR depot). This avoids local traps
    where a node looks good immediately but leads to a poor continuation.
    
    We refine this with a savings-like adjustment: we compare the "penalty" of extending
    the route to i and then taking the best next step, against the baseline of closing now.
    
    Score(i) = d(cur, i) + min( d(i, depot), min_{j valid} (d(i, j) + d(j, depot)) )
    
    We pick the i with minimum Score. This is equivalent to minimizing the 2-step cost.
    """
    # Time guard: if we are close to the limit, fall back to fastest heuristic (nearest)
    if _time_remaining() < 0.5:
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
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if len(feasible) == 0:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    # Convert to numpy array for vectorized distance calculation
    feasible_arr = np.array(feasible)
    n_feasible = len(feasible_arr)
    
    # Precompute distances from current to all feasible
    d_cur_to_f = distance_matrix[current_node, feasible_arr]
    
    # Precompute distance matrix among feasible nodes (subset)
    d_f_to_f = distance_matrix[np.ix_(feasible_arr, feasible_arr)]
    
    # Precompute distances from feasible to depot
    d_f_to_depot = distance_matrix[feasible_arr, depot]
    
    # Demands for feasible nodes
    demands_feasible = demands[feasible_arr]
    
    scores = np.zeros(n_feasible)
    
    for i_idx in range(n_feasible):
        node_i = feasible_arr[i_idx]
        cost_to_i = d_cur_to_f[i_idx]
        
        # Option 1: Go back to depot
        cost_back = d_f_to_depot[i_idx]
        
        # Option 2: Go to another node j
        # We need to find the best j.
        # Constraints:
        # 1. j != i
        # 2. j is in feasible_arr
        # 3. demands[j] <= rest_capacity - demands[i] (Capacity check for next step)
        
        best_next_cost = cost_back # Default to going back
        
        # Vectorized calculation for all j
        d_i_to_all = d_f_to_f[i_idx]
        d_all_to_depot = d_f_to_depot
        
        potential_costs = d_i_to_all + d_all_to_depot
        
        # Create a mask for valid j's
        valid_mask = np.ones(n_feasible, dtype=bool)
        valid_mask[i_idx] = False # Exclude self
        
        # Capacity check for next step
        remaining_cap_after_i = rest_capacity - demands[node_i]
        cap_mask = demands_feasible <= remaining_cap_after_i
        valid_mask &= cap_mask
        
        if np.any(valid_mask):
            # Find min cost among valid j's
            masked_costs = np.where(valid_mask, potential_costs, np.inf)
            min_next_cost = np.min(masked_costs)
            if min_next_cost < best_next_cost:
                best_next_cost = min_next_cost
        
        scores[i_idx] = cost_to_i + best_next_cost
    
    # Pick the node with the minimum score
    min_idx = np.argmin(scores)
    return int(feasible_arr[min_idx])

# EVOLVE-BLOCK-END
