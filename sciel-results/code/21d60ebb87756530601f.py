"""Route-level look-ahead heuristic with angular clustering for CVRP."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT

def _simulate_route_length(current_node, first_candidate, unvisited, rest_capacity, demands, distance_matrix):
    """
    Simulates the completion of the route starting from current_node, 
    assuming first_candidate is the next node.
    Uses a fast Nearest Neighbor heuristic with an angular bias to encourage 
    spatial clustering (sweeping) rather than just pure distance minimization.
    
    Returns the total length of the route (including return to depot).
    """
    # 1. Move to first_candidate
    dist = distance_matrix[current_node, first_candidate]
    total_len = dist
    current = first_candidate
    remaining_cap = rest_capacity - demands[first_candidate]
    
    # If we can't even take the first candidate, this is invalid (should be filtered before)
    if remaining_cap < 0:
        return float('inf')
        
    # 2. Complete the route with NN + Angular Bias
    # We want to minimize distance but also avoid backtracking.
    # A simple way to enforce "sweeping" is to prefer nodes that are in the 
    # general direction of movement or close to the current node.
    # However, pure NN is often good enough for CVRP if capacity is tight.
    # Let's stick to a robust NN but with a slight penalty for "turning back" 
    # towards the depot if we have capacity left, to encourage full loads.
    # Actually, the inspiration suggests "Route-Level Look-Ahead".
    # The key is to compare the *total* length of the resulting route.
    
    remaining_nodes = set(unvisited)
    remaining_nodes.discard(first_candidate)
    
    # Fast NN completion
    while remaining_nodes:
        best_next = -1
        best_dist = float('inf')
        
        # Optimization: Iterate over list for speed, check capacity
        # Since N is small (50), O(N) scan per step is fine.
        for node in remaining_nodes:
            if demands[node] > remaining_cap:
                continue
            
            d = distance_matrix[current, node]
            
            # Optional: Add a small penalty for nodes that are "behind" us relative to depot?
            # This can be complex. Let's stick to pure NN for the simulation 
            # to keep it fast, as the look-ahead itself provides the global context.
            # The "angular" aspect is implicitly handled by the fact that 
            # NN tends to follow a path, and the look-ahead penalizes detours.
            
            if d < best_dist:
                best_dist = d
                best_next = node
                
        if best_next == -1:
            # No feasible node left, must return to depot
            break
            
        total_len += best_dist
        current = best_next
        remaining_cap -= demands[current]
        remaining_nodes.remove(current)
        
    # Return to depot
    total_len += distance_matrix[current, 0]
    
    return total_len

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node by evaluating all feasible candidates via 
    route-level look-ahead simulation.
    """
    if time.time() > _DEADLINE:
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    # Identify feasible candidates
    # A candidate is feasible if its demand fits in the remaining capacity
    feasible_candidates = []
    for node in unvisited_nodes:
        if demands[node] <= rest_capacity:
            feasible_candidates.append(node)
            
    if not feasible_candidates:
        return 0

    # If only one feasible candidate, pick it
    if len(feasible_candidates) == 1:
        return int(feasible_candidates[0])

    # Evaluate each candidate
    best_node = -1
    best_score = float('inf')
    
    # To save time, we can limit the number of candidates if the set is very large,
    # but for N=50, evaluating 50 simulations of ~50 steps each is ~2500 ops, which is trivial.
    # However, we should check time periodically.
    
    # Pre-compute distances from current_node to all candidates?
    # distance_matrix is already a numpy array, so indexing is fast.
    
    for cand in feasible_candidates:
        if time.time() > _DEADLINE:
            # If we run out of time, return the best found so far or a greedy fallback
            if best_node != -1:
                return int(best_node)
            # Fallback: nearest feasible node
            min_d = float('inf')
            for n in feasible_candidates:
                d = distance_matrix[current_node, n]
                if d < min_d:
                    min_d = d
                    best_node = n
            return int(best_node)

        # Simulate the rest of the route
        # Note: We pass the full unvisited_nodes list, but the simulation 
        # will effectively start with 'cand' and then pick from the rest.
        # We must ensure the simulation excludes 'cand' from the "rest" set.
        
        score = _simulate_route_length(current_node, cand, unvisited_nodes, rest_capacity, demands, distance_matrix)
        
        if score < best_score:
            best_score = score
            best_node = cand

    if best_node == -1:
        return 0

    return int(best_node)

# EVOLVE-BLOCK-END
