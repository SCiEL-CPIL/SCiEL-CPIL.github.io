"""Route-completion look-ahead heuristic for CVRP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 115.0
_DEADLINE = _START_TIME + _TIMEOUT


def _complete_route_nn(current_node, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Fast nearest-neighbor completion of the current route.
    Returns the total additional distance to complete the route (including return to depot).
    """
    total_dist = 0.0
    cur = current_node
    remaining_cap = rest_capacity
    unvisited_set = set(unvisited_nodes)
    
    while unvisited_set:
        # Find feasible neighbors
        feasible = [n for n in unvisited_set if demands[n] <= remaining_cap]
        if not feasible:
            break
        
        # Pick nearest feasible node
        best_node = min(feasible, key=lambda n: distance_matrix[cur, n])
        total_dist += distance_matrix[cur, best_node]
        remaining_cap -= demands[best_node]
        cur = best_node
        unvisited_set.remove(best_node)
    
    # Return to depot
    total_dist += distance_matrix[cur, 0]
    return total_dist


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard
    if time.time() > _DEALINE if False else time.time() > _DEADLINE:
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return 0
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    # Convert to arrays for speed
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    if len(unvisited_arr) == 0:
        return 0

    demands_arr = np.array(demands, dtype=float)
    feasible_mask = demands_arr[unvisited_arr] <= rest_capacity

    if not np.any(feasible_mask):
        return 0

    feasible_nodes = unvisited_arr[feasible_mask]
    
    # If starting from depot, use nearest neighbor for first pick to establish direction
    if current_node == depot:
        dists_current = distance_matrix[current_node, feasible_nodes]
        best_idx = np.argmin(dists_current)
        return int(feasible_nodes[best_idx])
    
    # For subsequent nodes, use route-completion look-ahead
    # For each feasible candidate, estimate the total cost of completing the route
    # if we pick that candidate next.
    
    # Optimization: only do full look-ahead if the number of feasible nodes is small enough
    # Otherwise, fall back to a cheaper heuristic to stay within time budget
    n_feasible = len(feasible_nodes)
    
    if n_feasible <= 15:
        # Full look-ahead: simulate completion for each candidate
        best_score = float('inf')
        best_node = feasible_nodes[0]
        
        for cand in feasible_nodes:
            # Cost to go from current to candidate
            cost_to_cand = distance_matrix[current_node, cand]
            
            # Remaining capacity after visiting candidate
            new_rest_cap = rest_capacity - demands[cand]
            
            # Remaining unvisited nodes (excluding candidate)
            remaining_unvisited = [n for n in unvisited_nodes if n != cand]
            
            # Simulate completion from candidate
            completion_cost = _complete_route_nn(cand, remaining_unvisited, new_rest_cap, demands, distance_matrix)
            
            total_cost = cost_to_cand + completion_cost
            
            if total_cost < best_score:
                best_score = total_cost
                best_node = cand
        
        return int(best_node)
    else:
        # Too many candidates for full look-ahead; use a hybrid scoring
        # Combine nearest-neighbor distance with a savings heuristic
        dists_current = distance_matrix[current_node, feasible_nodes]
        dists_depot = distance_matrix[feasible_nodes, depot]
        
        # Savings-based score: prefer nodes that are close to current but far from depot
        alpha = 0.8
        scores = dists_current - alpha * dists_depot
        
        # Add demand urgency: higher demand nodes are more urgent
        demand_arr = demands_arr[feasible_nodes]
        max_demand = demand_arr.max() if len(demand_arr) > 0 else 1.0
        if max_demand > 0:
            demand_urgency = demand_arr / max_demand
            scores = scores - 0.1 * demand_urgency
        
        best_idx = np.argmin(scores)
        return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
