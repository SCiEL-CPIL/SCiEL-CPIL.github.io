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
    3. Score each feasible candidate using a combination of:
       - Savings-based metric (Clarke-Wright): dist(depot, current) + dist(depot, j) - dist(current, j)
       - Angular/detour consistency: how aligned j is with the current direction from depot
       - 1-step lookahead: min distance from candidate to other feasible nodes or depot
       - Capacity-based penalty to encourage packing
    4. If time is running out, fall back to nearest feasible node.
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
    
    # Savings-based metric: dist(depot, current) + dist(depot, j) - dist(current, j)
    # Higher savings means j is "cheaper" to append than serving it separately
    dist_depot_current = distance_matrix[depot, current_node]
    dist_depot_feasible = distance_matrix[depot, feasible_arr]
    savings = dist_depot_current + dist_depot_feasible - immediate
    
    # Angular/detour consistency: how aligned is j with the direction from depot through current?
    # detour = dist(depot, current) + dist(current, j) - dist(depot, j)
    # Small detour means j is roughly in the same direction as current from depot
    detour = dist_depot_current + immediate - dist_depot_feasible
    
    # 1-step lookahead: for each candidate, min distance to any other feasible node or depot
    dist_feas = distance_matrix[np.ix_(feasible_arr, feasible_arr)]
    np.fill_diagonal(dist_feas, np.inf)
    min_to_other_feas = np.min(dist_feas, axis=1)
    
    # Distance from each feasible node to depot
    dist_to_depot = distance_matrix[feasible_arr, depot]
    
    # Lookahead: min of going to another feasible or going to depot
    lookahead = np.minimum(min_to_other_feas, dist_to_depot)
    
    # Capacity penalty: penalize nodes that consume a large fraction of remaining capacity
    cap_penalty = demands[feasible_arr] / max(rest_capacity, 1)
    
    # Normalize each component to [0, 1] for weighted combination
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / rng
    
    # We want to maximize savings, minimize detour, minimize lookahead, minimize cap_penalty
    savings_norm = normalize(savings)
    detour_norm = normalize(detour)  # smaller is better
    lookahead_norm = normalize(lookahead)  # smaller is better
    cap_norm = normalize(cap_penalty)  # smaller is better
    
    # Weighted composite score (higher is better)
    # Savings is the dominant signal from Clarke-Wright
    # Detour/angular consistency helps maintain route coherence
    # Lookahead avoids dead-ends
    # Capacity penalty encourages balanced loading
    w_savings = 0.45
    w_detour = 0.25
    w_lookahead = 0.20
    w_cap = 0.10
    
    composite = (w_savings * savings_norm 
                 + w_detour * (1.0 - detour_norm) 
                 + w_lookahead * (1.0 - lookahead_norm) 
                 + w_cap * (1.0 - cap_norm))
    
    # Pick the candidate with the highest composite score
    best_idx = int(np.argmax(composite))
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
