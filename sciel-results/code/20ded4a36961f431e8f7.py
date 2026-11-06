import numpy as np
import time

_START = time.time()
_LIMIT = 150.0 - 15.0  # seconds of compute budget
_CAPACITY = 50.0


def _estimate_remaining_cost(start_node, current_cap, unvisited_list, dm, demands):
    """
    Estimate the cost to complete the route starting from start_node,
    visiting a subset of unvisited_list such that capacity is respected,
    and returning to depot (0).
    Uses a greedy nearest-neighbor strategy for the simulation.
    """
    if not unvisited_list:
        return dm[start_node, 0]
    
    current = start_node
    remaining_cap = current_cap
    # Use a list of remaining nodes to visit
    remaining = list(unvisited_list)
    total_cost = 0.0
    
    while remaining:
        # Find nearest feasible node
        best_next = -1
        best_dist = float('inf')
        
        # Optimization: If few nodes left, we can check all. 
        # For N=50, this inner loop is small.
        for i, n in enumerate(remaining):
            if demands[n] <= remaining_cap:
                d = dm[current, n]
                if d < best_dist:
                    best_dist = d
                    best_next = n
                    best_idx = i
        
        if best_next == -1:
            # No more nodes fit, return to depot
            total_cost += dm[current, 0]
            break
            
        total_cost += best_dist
        remaining_cap -= demands[best_next]
        current = best_next
        # Remove the visited node
        remaining.pop(best_idx)
        
    return total_cost

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    t0 = time.time()
    deadline = _START + _LIMIT

    if len(unvisited_nodes) == 0:
        return depot

    unvisited = list(unvisited_nodes)

    # Feasible nodes: demand <= rest_capacity
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    
    if not feasible:
        return depot
        
    if len(feasible) == 1:
        return feasible[0]

    dm = np.asarray(distance_matrix)
    
    best_node = -1
    best_score = float('inf')
    
    for n in feasible:
        if time.time() > deadline:
            break
            
        # 1. Cost to get to n
        cost_to_n = dm[current_node, n]
        
        # 2. Estimated cost to finish route from n
        # We pass a list of unvisited nodes excluding n
        # Creating a new list for each candidate is O(N), total O(N^2) for this part.
        remaining_unvisited = [u for u in unvisited if u != n]
        
        # We also need to update capacity.
        new_cap = rest_capacity - demands[n]
        
        sim_cost = _estimate_remaining_cost(n, new_cap, remaining_unvisited, dm, demands)
        
        total_est_cost = cost_to_n + sim_cost
        
        if total_est_cost < best_score:
            best_score = total_est_cost
            best_node = n
            
    if best_node == -1:
        # Fallback if timeout hit before any candidate was scored
        best_node = feasible[0]
        
    return int(best_node)
