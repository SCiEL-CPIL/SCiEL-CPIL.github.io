import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0
_DEADLINE = _start_time + _TIMEOUT

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if time.time() > _DEADLINE:
        return 0
    
    if len(unvisited_nodes) == 0:
        return 0
    
    unvisited = np.array(unvisited_nodes, dtype=int)
    
    # Filter by capacity
    feasible_mask = np.array([demands[n] <= rest_capacity for n in unvisited])
    feasible = unvisited[feasible_mask]
    
    if len(feasible) == 0:
        return 0
    
    # Compute savings for each feasible node
    # savings(j) = dist(current, depot) + dist(depot, j) - dist(current, j)
    # This represents the route length saved by visiting j now vs. going to depot first
    dist_current_depot = distance_matrix[current_node, depot]
    dist_depot_feasible = distance_matrix[depot, feasible]
    dist_current_feasible = distance_matrix[current_node, feasible]
    
    savings = dist_current_depot + dist_depot_feasible - dist_current_feasible
    
    # If the best savings is non-positive, it's better to close the route
    if np.max(savings) <= 0:
        return 0
    
    # Capacity-aware penalty: discourage choices that leave very little capacity
    remaining_after = rest_capacity - np.array([demands[n] for n in feasible], dtype=float)
    
    # Compute the minimum demand among all unvisited nodes
    all_demands = np.array([demands[n] for n in unvisited])
    min_demand = np.min(all_demands)
    avg_demand = np.mean(all_demands)
    
    # Penalty: if remaining capacity after visiting j is less than average demand,
    # we'll likely have to go to depot soon, so penalize
    penalty = 3.0 * np.maximum(0.0, avg_demand - remaining_after)
    
    # Bonus for nodes far from depot (they're expensive to serve in a new route)
    max_depot_dist = np.max(dist_depot_feasible)
    if max_depot_dist > 0:
        far_bonus = 0.15 * dist_depot_feasible / max_depot_dist
    else:
        far_bonus = np.zeros(len(feasible))
    
    scores = savings - penalty + far_bonus
    
    best_idx = np.argmax(scores)
    return int(feasible[best_idx])
