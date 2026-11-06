"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 2.0

def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated vehicle routing:
    Pick the next customer to visit, or return to depot (0) to close the route.
    
    Strategy:
    1. Filter unvisited nodes that fit within rest_capacity.
    2. If none fit, return depot.
    3. For each feasible candidate j, estimate the total additional cost:
       cost(j) = dist(current, j) + min( dist(j, depot), min_{k in feasible_from_j} (dist(j, k) + dist(k, depot)) )
       This is a 2-step lookahead that considers the cost of finishing the route.
    4. Pick the candidate with the minimum estimated cost.
    5. If time is running out, fall back to nearest feasible node.
    """
    if _time_exceeded():
        # Fallback: nearest feasible node
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return int(depot)
        if len(feasible) == 1:
            return int(feasible[0])
        dists = distance_matrix[current_node, feasible]
        return int(feasible[int(np.argmin(dists))])

    # Filter feasible nodes
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if not feasible:
        return int(depot)
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    feasible_arr = np.array(feasible)
    n_feas = len(feasible_arr)
    
    # Immediate distance from current node to each feasible candidate
    immediate = distance_matrix[current_node, feasible_arr]
    
    # Distance from each feasible node to depot
    dist_to_depot = distance_matrix[feasible_arr, depot]
    
    # For each candidate j, we need to find the best next move k.
    # k must be in unvisited_nodes (excluding j) and demands[k] <= rest_capacity - demands[j]
    
    # To speed this up, we can pre-compute the min dist(k, depot) for all k.
    # But the feasibility of k depends on j's demand.
    
    # Let's iterate through feasible candidates
    best_score = np.inf
    best_idx = 0
    
    # Pre-compute distances from feasible nodes to all other nodes?
    # distance_matrix is available.
    
    # We need min_{k in feasible_from_j} (dist(j, k) + dist(k, depot))
    # Let's compute this for each j in feasible_arr
    
    # To avoid O(N^2) per call if N is large, but N=50 is small.
    # O(N^2) is 2500 ops, which is fast.
    
    for idx, j in enumerate(feasible_arr):
        # Immediate cost
        cost_direct = immediate[idx]
        
        # Remaining capacity after visiting j
        rem_cap_j = rest_capacity - demands[j]
        
        # Find nodes k that can be visited after j
        # k must be in unvisited_nodes, k != j, and demands[k] <= rem_cap_j
        # We can filter unvisited_nodes again or use the feasible list if we are careful.
        # Actually, feasible nodes are those that fit in rest_capacity.
        # Nodes that fit in rem_cap_j are a subset of feasible nodes (since rem_cap_j <= rest_capacity).
        # So we only need to look at feasible_arr.
        
        # Get indices of nodes in feasible_arr that fit in rem_cap_j
        # demands[feasible_arr] <= rem_cap_j
        # Also exclude j itself (index idx)
        
        # Mask for feasible nodes that fit in rem_cap_j
        # demands is a dict or array? "demands" in signature. 
        # In the seed, demands[n] is used, so it's likely a dict or array.
        # Assuming it's array-like for numpy indexing.
        
        # If demands is a dict, we need to handle it.
        # The seed uses demands[n]. Let's assume it's a list or array.
        
        # Let's get demands for feasible_arr
        # If demands is a dict, this fails. But seed uses demands[feasible_arr].
        # So demands must be array-like.
        
        feas_demands = demands[feasible_arr]
        
        # Mask: demands[k] <= rem_cap_j
        mask = feas_demands <= rem_cap_j
        # Exclude j itself
        mask[idx] = False
        
        if np.any(mask):
            # Get indices of valid k's
            valid_k_indices = np.where(mask)[0]
            
            # Distances from j to these k's
            # distance_matrix[j, feasible_arr[valid_k_indices]]
            dists_j_to_k = distance_matrix[j, feasible_arr[valid_k_indices]]
            
            # Distances from these k's to depot
            # distance_matrix[feasible_arr[valid_k_indices], depot]
            dists_k_to_depot = distance_matrix[feasible_arr[valid_k_indices], depot]
            
            # Total cost for path j -> k -> depot
            costs_j_k_depot = dists_j_to_k + dists_k_to_depot
            
            # Min cost among these
            min_lookahead = np.min(costs_j_k_depot)
        else:
            # No other node can be visited after j
            # So we must go to depot
            min_lookahead = np.inf
        
        # The actual lookahead cost is the min of going directly to depot or going via k
        # cost_finish = min(dist(j, depot), min_lookahead)
        cost_finish = min(dist_to_depot[idx], min_lookahead)
        
        total_cost = cost_direct + cost_finish
        
        if total_cost < best_score:
            best_score = total_cost
            best_idx = idx
            
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
