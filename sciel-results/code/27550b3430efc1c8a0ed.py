"""Route-level look-ahead heuristic for Capacitated Vehicle Routing Problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT

def _complete_route_nn(current_node, depot, unvisited, rest_capacity, demands, distance_matrix):
    """
    Greedily complete the route from current_node using Nearest Neighbor.
    Returns the total length of the completed route including the return to depot.
    """
    total_len = 0.0
    current = current_node
    remaining_cap = rest_capacity
    remaining = set(unvisited)
    
    # Optimization: If few nodes left, exact or simple NN is fine.
    # If many nodes, this loop might be slow if called for every candidate.
    # We limit the depth of this simulation if the set is large to stay within time.
    # However, for 50 customers, even a full NN simulation for ~10 candidates is ~500 ops.
    # 500 ops * 50 steps * 8 instances is trivial.
    
    while remaining:
        # Find nearest feasible neighbor
        best_next = -1
        best_dist = float('inf')
        
        # Iterate over remaining nodes
        # Using a list comprehension or loop is fine for small N
        for node in remaining:
            if demands[node] > remaining_cap:
                continue
            d = distance_matrix[current, node]
            if d < best_dist:
                best_dist = d
                best_next = node
        
        if best_next == -1:
            break
            
        total_len += best_dist
        current = best_next
        remaining_cap -= demands[current]
        remaining.remove(current)
        
    # Return to depot
    total_len += distance_matrix[current, depot]
    
    return total_len

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node by simulating the rest of the route for each feasible candidate
    and choosing the one that minimizes the total completed route length.
    """
    if time.time() > _DEADLINE:
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    # 1. Filter by capacity
    unvisited = np.array(unvisited_nodes)
    feasible = unvisited[demands[unvisited] <= rest_capacity]
    
    if len(feasible) == 0:
        return 0

    # If only one feasible, return it
    if len(feasible) == 1:
        return int(feasible[0])

    # 2. For each feasible candidate, estimate the total route length if we pick it next
    best_node = int(feasible[0])
    best_score = float('inf')
    
    # To save time, we can limit the look-ahead or just do it for all if N is small.
    # With N=50, the inner loop is small.
    
    for cand in feasible:
        # Simulate: current -> cand -> (NN completion from cand) -> depot
        # Cost = dist(current, cand) + cost_of_NN_completion_from_cand
        
        d_first = distance_matrix[current_node, cand]
        
        # We need to simulate NN from 'cand' with 'cand' removed from unvisited
        # and capacity reduced by demands[cand]
        
        # Optimization: Instead of a full recursive call or complex set ops,
        # we can just run the NN logic inline or call a helper.
        # The helper _complete_route_nn above takes current_node and unvisited.
        # We need to pass (cand, unvisited - {cand}, rest_capacity - demands[cand])
        
        # To avoid creating new sets/arrays repeatedly which is slow in Python,
        # let's inline a simplified NN completion logic specifically for this evaluation.
        
        remaining_set = set(unvisited_nodes)
        remaining_set.discard(cand)
        rem_cap = rest_capacity - demands[cand]
        
        sim_total = d_first
        cur = cand
        
        # Fast NN loop
        # We can break early if sim_total already exceeds best_score + lower_bound_return
        lower_bound_return = distance_matrix[cur, depot]
        
        while remaining_set:
            # Find nearest feasible neighbor
            best_next = -1
            best_dist = float('inf')
            
            for node in remaining_set:
                if demands[node] > rem_cap:
                    continue
                d = distance_matrix[cur, node]
                if d < best_dist:
                    best_dist = d
                    best_next = node
            
            if best_next == -1:
                break
                
            sim_total += best_dist
            cur = best_next
            rem_cap -= demands[cur]
            remaining_set.remove(cur)
            
            # Pruning: if current sim_total + min possible return is already worse than best_score
            # min possible return is roughly distance_matrix[cur, depot]
            if sim_total + distance_matrix[cur, depot] >= best_score:
                break
        
        sim_total += distance_matrix[cur, depot]
        
        if sim_total < best_score:
            best_score = sim_total
            best_node = int(cand)
            
    return best_node

# EVOLVE-BLOCK-END
