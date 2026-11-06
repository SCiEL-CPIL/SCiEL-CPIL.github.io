"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track elapsed time to ensure we finish within the 150s limit
_start_time = time.time()
_DEADLINE = 150.0
_TIME_MARGIN = 5.0  # safety margin in seconds


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: pick next customer or depot (0) to close route.
    Strategy:
      1. Filter unvisited nodes that fit in rest_capacity.
      2. If none fit, return depot.
      3. Score each feasible node by a combination of:
         - immediate distance from current_node
         - distance from that node to depot (to encourage closing routes when capacity is low)
         - a lookahead term: minimum distance from that node to any other feasible node
    """
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
    
    # Lookahead weight: slight preference for nodes that keep options open
    lookahead_weight = 0.2
    
    best_node = None
    best_score = float('inf')
    
    for n in feasible:
        dist_current = distance_matrix[current_node, n]
        dist_depot = distance_matrix[n, depot]
        
        # Lookahead: minimum distance from n to any other feasible node
        # This encourages picking nodes that don't isolate themselves
        if len(feasible) > 1:
            other_feasible = [m for m in feasible if m != n]
            min_lookahead = min(distance_matrix[n, m] for m in other_feasible)
        else:
            min_lookahead = 0.0
        
        score = (1.0 - depot_weight - lookahead_weight) * dist_current + \
                depot_weight * dist_depot + \
                lookahead_weight * min_lookahead
        
        if score < best_score:
            best_score = score
            best_node = n
    
    return int(best_node)
# EVOLVE-BLOCK-END
