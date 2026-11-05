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
    Selects the next node using a 2-step lookahead heuristic.
    For each feasible node i, we estimate the cost of:
    1. Going from current to i.
    2. Going from i to the best subsequent node j (considering capacity and distance) OR back to depot.
    
    Score(i) = d(cur, i) + min( d(i, depot), min_{j in feasible\{i}, cap_ok} (d(i, j) + d(j, depot)) )
    
    This structural change allows the heuristic to avoid "traps" where visiting a node
    leads to a high-cost return to depot or leaves no good neighbors, whereas a slightly
    worse immediate choice might lead to a tighter cluster.
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
    # This is O(N^2) but N is small (<=50)
    d_f_to_f = distance_matrix[np.ix_(feasible_arr, feasible_arr)]
    
    # Precompute distances from feasible to depot
    d_f_to_depot = distance_matrix[feasible_arr, depot]
    
    # We need to evaluate the score for each candidate i in feasible_arr.
    # Score(i) = d_cur_to_f[i] + FutureCost(i)
    # FutureCost(i) is the min cost to complete the route from i.
    # Approximation: min( d(i, depot), min_{j != i, feasible} (d(i, j) + d(j, depot)) )
    # Note: We must ensure that if we go to j, j is feasible from i's perspective?
    # The interface only gives rest_capacity for the CURRENT state.
    # Usually, rest_capacity decreases as we visit nodes.
    # However, the function signature implies we only know the current rest_capacity.
    # We don't know the capacity remaining after visiting i.
    # But we can assume that if j is in the current feasible set, it might not be feasible after i.
    # To be safe and simple, we consider j as a potential next step if it is in the current feasible set.
    # A stricter check would be demands[j] <= rest_capacity - demands[i].
    # Let's add this capacity check for the lookahead step to be more accurate.
    
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
        # 3. demands[j] <= rest_capacity - demands[node_i] (Capacity check for next step)
        
        best_next_cost = cost_back # Default to going back
        
        # Vectorized calculation for all j
        # We can compute d(i, j) + d(j, depot) for all j
        # Then mask out j where capacity is violated or j == i
        
        d_i_to_all = d_f_to_f[i_idx]
        d_all_to_depot = d_f_to_depot
        
        potential_costs = d_i_to_all + d_all_to_depot
        
        # Mask:
        # 1. j != i (index i_idx)
        # 2. demands[j] <= rest_capacity - demands[i]
        
        # Create a mask for valid j's
        valid_mask = np.ones(n_feasible, dtype=bool)
        valid_mask[i_idx] = False # Exclude self
        
        # Capacity check for next step
        remaining_cap_after_i = rest_capacity - demands[node_i]
        demands_feasible = demands[feasible_arr]
        cap_mask = demands_feasible <= remaining_cap_after_i
        valid_mask &= cap_mask
        
        if np.any(valid_mask):
            # Find min cost among valid j's
            # We can use np.where to set invalid ones to inf
            masked_costs = np.where(valid_mask, potential_costs, np.inf)
            min_next_cost = np.min(masked_costs)
            if min_next_cost < best_next_cost:
                best_next_cost = min_next_cost
        
        scores[i_idx] = cost_to_i + best_next_cost
    
    # Pick the node with the minimum score
    min_idx = np.argmin(scores)
    return int(feasible_arr[min_idx])

# EVOLVE-BLOCK-END
