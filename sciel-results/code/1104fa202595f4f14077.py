"""Route-Level Look-Ahead Heuristic for Capacitated Vehicle Routing Problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT

def _complete_route_nn(start, remaining_cap, unvisited_set, demands, dist, depot):
    """
    Fast nearest-neighbor-with-capacity completion of a route.
    Returns the total extra distance to complete the route from `start`.
    This is used as a proxy for the "future cost" of a decision.
    """
    total = 0.0
    cur = start
    cap = remaining_cap
    remaining = list(unvisited_set)
    
    # Optimization: If the set is small, list operations are fine. 
    # For larger sets, we could use a heap, but NN is O(N^2) anyway.
    while remaining:
        # Find feasible nodes
        # In this context, unvisited_set is the set of nodes NOT yet visited in the current route simulation.
        # Note: The caller passes the set of nodes that are *candidates* for the rest of the route.
        # However, for a true look-ahead, we should simulate adding the *rest* of the unvisited nodes 
        # that fit in the remaining capacity.
        
        # Find the nearest feasible node
        best_n = -1
        best_d = np.inf
        for n in remaining:
            if demands[n] <= cap:
                d = dist[cur, n]
                if d < best_d:
                    best_d = d
                    best_n = n
        
        if best_n == -1:
            # No more feasible nodes fit. We must return to depot.
            total += dist[cur, depot]
            break
        
        # Move to best_n
        total += best_d
        cur = best_n
        cap -= demands[cur]
        remaining.remove(cur)
        
    # If we exited the loop because remaining was empty, we still need to go back to depot
    # But wait, the loop breaks if no feasible nodes. 
    # If remaining is empty, the loop terminates naturally.
    # We must always return to depot at the end of the route.
    if remaining:
        # We broke out due to capacity. Go back to depot.
        total += dist[cur, depot]
    else:
        # We finished all nodes. Go back to depot.
        total += dist[cur, depot]
        
    return total

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a Route-Level Look-Ahead heuristic.
    
    Strategy:
    1. Identify all feasible unvisited nodes (demand <= rest_capacity).
    2. For each candidate node C:
       a. Simulate the rest of the route starting from C, using the remaining capacity 
          after visiting C and the set of unvisited nodes excluding C.
       b. Use a fast Nearest-Neighbor with Capacity (NNC) to estimate the total cost 
          of completing this route.
       c. Total Cost(C) = dist(current, C) + EstimatedCompletionCost(C)
    3. Select the candidate C that minimizes Total Cost(C).
    4. If no feasible nodes, return depot.
    
    This looks beyond the immediate step to evaluate the quality of the entire route 
    that would result from picking C.
    """
    if time.time() > _DEADLINE:
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    unvisited = np.array(unvisited_nodes)
    
    # 1. Filter by capacity
    feasible = unvisited[demands[unvisited] <= rest_capacity]
    
    if len(feasible) == 0:
        return 0

    best_node = -1
    best_score = np.inf
    
    # To speed up, if the number of feasible nodes is large, we might need to be careful.
    # However, 50 nodes is small. NNC is O(N^2) per candidate. 
    # Total complexity: O(|Feasible| * N^2). With N=50, this is 50 * 2500 = 125,000 ops per step.
    # 50 steps total -> 6.25M ops. This is well within the time limit in Python/NumPy.
    
    for cand in feasible:
        # Cost to reach candidate
        cost_to_cand = distance_matrix[current_node, cand]
        
        # Remaining capacity after visiting candidate
        rem_cap_after = rest_capacity - demands[cand]
        
        # Set of unvisited nodes excluding the candidate
        # We need to simulate the rest of the route with the *other* unvisited nodes.
        # Note: We should only consider nodes that *could* potentially fit? 
        # The NNC simulation will handle capacity constraints dynamically.
        other_unvisited = [n for n in unvisited if n != cand]
        
        # Simulate completion
        # We pass the list of other unvisited nodes. 
        # The NNC function will pick nodes from this list that fit in rem_cap_after.
        completion_cost = _complete_route_nn(cand, rem_cap_after, other_unvisited, demands, distance_matrix, depot)
        
        total_cost = cost_to_cand + completion_cost
        
        if total_cost < best_score:
            best_score = total_cost
            best_node = cand
            
        # Time check inside loop for safety
        if time.time() > _DEADLINE:
            break

    if best_node == -1:
        return 0
        
    return int(best_node)

# EVOLVE-BLOCK-END
