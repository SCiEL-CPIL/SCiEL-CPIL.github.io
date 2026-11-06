"""Improved heuristic for Capacitated VRP with capacity awareness."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: track start time to avoid timeout
_START_TIME = None
_TIME_LIMIT = 115.0  # Safety margin before 120s kill

def _check_deadline():
    """Return True if we should stop and return depot for safety."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) > _TIME_LIMIT


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Score each feasible customer by:
      - distance from current node
      - estimated additional cost to return to depot after visiting
      - capacity utilization efficiency (prefer customers that use remaining capacity well)
    
    Returns 0 (depot) to close route if no feasible customers or deadline exceeded.
    """
    # Deadline guard
    if _check_deadline():
        return 0
    
    if len(unvisited_nodes) == 0:
        return 0
    
    unvisited_arr = np.array(unvisited_nodes)
    
    # Filter to only feasible customers (demand <= rest_capacity)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return 0
    
    # Get distances from current node to feasible customers
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    
    # Get distances from feasible customers back to depot
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Total route extension if we visit this customer:
    # current -> customer -> depot (if this were the last)
    # But we also need to consider that we might visit more customers after.
    # Heuristic: score = dist_from_current + alpha * dist_to_depot
    
    # Capacity-aware scoring:
    # Prefer customers that leave enough capacity for more visits
    # but don't waste too much capacity on a single small demand far away
    
    demands_feasible = demands[feasible_nodes]
    
    # Base score: travel cost
    # We want to minimize: dist_from_current + lambda * dist_to_depot
    # This approximates the incremental cost of adding this customer to the route
    
    # Additionally, penalize customers with very small demand relative to remaining capacity
    # if they are far, as they waste capacity
    
    # Weighted score
    lambda_return = 0.5  # Weight for return-to-depot distance
    
    # Score: lower is better
    scores = dist_from_current + lambda_return * dist_to_depot
    
    # Capacity efficiency penalty:
    # If demand is small relative to rest_capacity, and distance is high, penalize
    # This encourages packing more demand per unit distance
    cap_ratio = demands_feasible / float(rest_capacity)
    # Small penalty for low capacity utilization (encourages larger demands first)
    # But only if the node is not very close
    avg_dist = np.mean(dist_from_current)
    if avg_dist > 0:
        # Normalize distances to [0, 1] range for penalty scaling
        dist_norm = dist_from_current / (avg_dist * 2 + 1e-9)
        # Penalize low cap_ratio when distance is high
        cap_penalty = 0.1 * (1 - cap_ratio) * dist_norm
        scores = scores + cap_penalty
    
    # Pick the customer with minimum score
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
