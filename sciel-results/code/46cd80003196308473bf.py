"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track elapsed time to ensure we finish within the 150s limit
_start_time = time.time()
_DEADLINE = 150.0
_TIME_MARGIN = 5.0  # safety margin in seconds


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard: if we're running low on time, fall back to simple nearest feasible
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _TIME_MARGIN:
        # Fallback: nearest feasible customer
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        return int(min(feasible, key=lambda n: distance_matrix[current_node, n]))
    
    # Find feasible nodes (can be visited within capacity)
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if not feasible:
        return depot
    
    # If only one feasible node, return it
    if len(feasible) == 1:
        return feasible[0]
    
    # Heuristic: balance proximity to current node with proximity to depot
    # This helps close routes earlier when remaining capacity is low
    # Score: combination of distance from current and distance to depot
    # Weight the depot distance more heavily when remaining capacity is low
    
    remaining_ratio = rest_capacity / 50.0  # normalize by max capacity
    
    # When capacity is low, prefer nodes closer to depot to close route sooner
    # When capacity is high, prefer nearest neighbor
    depot_weight = max(0.0, 1.0 - remaining_ratio) * 0.5  # up to 0.5 weight on depot proximity
    
    best_node = None
    best_score = float('inf')
    
    for n in feasible:
        dist_current = distance_matrix[current_node, n]
        dist_depot = distance_matrix[n, depot]
        score = (1.0 - depot_weight) * dist_current + depot_weight * dist_depot
        if score < best_score:
            best_score = score
            best_node = n
    
    return int(best_node)
# EVOLVE-BLOCK-END
