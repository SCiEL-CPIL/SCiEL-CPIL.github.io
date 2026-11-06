import numpy as np
import time

_START = time.time()
_LIMIT = 150.0 - 15.0  # seconds of compute budget
_CAPACITY = 50.0


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
    
    # Adaptive weight for returning to depot
    # High capacity -> low weight (keep exploring)
    # Low capacity -> high weight (return to depot)
    w = 1.0 - (rest_capacity / _CAPACITY)
    
    # Packing bonus: prefer higher demand nodes to fill the truck
    alpha = 0.1
    
    best_node = -1
    best_score = float('inf')
    
    for n in feasible:
        if time.time() > deadline:
            break
        d_cur = dm[current_node, n]
        d_dep = dm[n, depot]
        demand_bonus = alpha * demands[n]
        
        # Score: immediate distance + weighted return distance - packing bonus
        score = d_cur + w * d_dep - demand_bonus
        
        if score < best_score:
            best_score = score
            best_node = n
            
    if best_node == -1:
        # Fallback if timeout hit before any candidate was scored
        best_node = feasible[0]
        
    return int(best_node)
