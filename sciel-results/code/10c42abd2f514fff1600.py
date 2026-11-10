"""Hybrid route-level look-ahead heuristic for Capacitated Vehicle Routing Problem."""
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
    """
    total = 0.0
    cur = start
    cap = remaining_cap
    remaining = list(unvisited_set)
    
    while remaining:
        feasible = [n for n in remaining if demands[n] <= cap]
        if not feasible:
            break
        best_n = None
        best_d = np.inf
        for n in feasible:
            d = dist[cur, n]
            if d < best_d:
                best_d = d
                best_n = n
        total += best_d
        cap -= demands[best_n]
        cur = best_n
        remaining.remove(best_n)
        if cap <= 0:
            break
            
    total += dist[cur, depot]
    return total

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a hybrid strategy:
    1. If the number of feasible candidates is small (<= 15), perform a full route-level
       look-ahead simulation (Nearest Neighbor completion) to minimize predicted total route length.
    2. Otherwise, use an adaptive savings/distance blend to make a fast greedy choice.
    
    This combines the global structure awareness of the look-ahead approach with the
    speed and robustness of the adaptive greedy method.
    """
    if time.time() > _DEADLINE:
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    unvisited = np.array(unvisited_nodes)
    dist = distance_matrix
    
    # Filter by capacity
    feasible = unvisited[demands[unvisited] <= rest_capacity]
    
    if len(feasible) == 0:
        return 0

    # Strategy 1: Route-level Look-ahead for small feasible sets
    # This is expensive (O(N^2) per candidate), so only use when N is small.
    # For 50 customers, early in the route N is large, so we fall back to Strategy 2.
    # Later in the route, N is small, so we get the precise global optimization.
    
    if len(feasible) <= 15:
        best_c = None
        best_score = np.inf
        feasible_list = feasible.tolist()
        
        for c in feasible_list:
            if time.time() > _DEADLINE:
                break
            d_first = dist[current_node, c]
            new_cap = rest_capacity - demands[c]
            remaining_for_completion = [n for n in feasible_list if n != c]
            extra = _complete_route_nn(c, new_cap, remaining_for_completion, demands, dist, depot)
            score = d_first + extra
            if score < best_score:
                best_score = score
                best_c = c
                
        if best_c is not None:
            return int(best_c)
        # Fallback if time ran out during simulation
        best_c = min(feasible_list, key=lambda n: dist[current_node, n])
        return int(best_c)

    # Strategy 2: Adaptive Savings/Distance Blend for larger feasible sets
    # This is O(N) and fast.
    
    d_cur_dep = dist[current_node, depot]
    d_dep_cand = dist[depot, feasible]
    d_cur_cand = dist[current_node, feasible]
    
    # Standard Savings
    savings = d_cur_dep + d_dep_cand - d_cur_cand
    
    # Normalize Savings
    s_min = np.min(savings)
    s_max = np.max(savings)
    if s_max == s_min:
        norm_savings = np.zeros_like(savings, dtype=float)
    else:
        norm_savings = (savings - s_min) / (s_max - s_min)
        
    # Normalize Distance (inverted: closer is better)
    dist_to_cur = d_cur_cand
    dist_min = np.min(dist_to_cur)
    dist_max = np.max(dist_to_cur)
    if dist_max == dist_min:
        norm_dist = np.zeros_like(dist_to_cur, dtype=float)
    else:
        norm_dist = (dist_max - dist_to_cur) / (dist_max - dist_min)
        
    # Adaptive Alpha based on remaining capacity
    # High capacity -> more freedom to optimize for savings (global structure)
    # Low capacity -> prioritize closeness (safety/compactness)
    CAPACITY = 50.0
    cap_ratio = rest_capacity / CAPACITY
    
    if cap_ratio > 0.6:
        alpha = 0.8  # More savings
    elif cap_ratio > 0.3:
        alpha = 0.6  # Balanced
    else:
        alpha = 0.4  # More distance (safety)
        
    combined_score = alpha * norm_savings + (1 - alpha) * norm_dist
    
    best_idx = np.argmax(combined_score)
    
    return int(feasible[best_idx])

# EVOLVE-BLOCK-END
