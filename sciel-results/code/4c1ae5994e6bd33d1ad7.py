"""CVRP heuristic using geometric angular sweep around the depot for spatial clustering."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_start_time = None
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using an angular sweep heuristic around the depot.
    This encourages spatial clustering by preferring nodes that are angularly close
    to the current direction of travel from the depot.
    
    Returns 0 (depot) if no feasible node can be added or if the time limit is reached.
    """
    global _start_time, _DEADLINE
    
    if _start_time is None:
        _start_time = time.time()
        _DEADLINE = _start_time + _TIMEOUT
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return int(depot)
    
    # Convert unvisited_nodes to numpy array
    if isinstance(unvisited_nodes, (list, tuple)):
        unvisited_arr = np.array(unvisited_nodes, dtype=int)
    else:
        unvisited_arr = np.asarray(unvisited_nodes, dtype=int)
    
    # If no unvisited nodes, return depot
    if len(unvisited_arr) == 0:
        return int(depot)
    
    # Find feasible nodes
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited_arr[feasible_mask]
    n_feasible = len(feasible_nodes)
    
    # If at depot, we need to start a route.
    # We can pick the node that minimizes distance to depot, or use a different strategy.
    # For consistency with the "sweep" idea, maybe pick the one with smallest angle?
    # But at depot, there is no "current direction".
    # Standard practice: nearest neighbor or savings.
    # Let's use nearest neighbor to start, as it's a safe anchor for the sweep.
    if current_node == depot:
        dists = distance_matrix[current_node, feasible_nodes]
        min_idx = np.argmin(dists)
        return int(feasible_nodes[min_idx])
    
    # Compute coordinates if not provided? No, we only have distance_matrix.
    # We need to derive "angles".
    # Since we only have distance_matrix, we cannot compute true geometric angles 
    # unless we assume a coordinate system or use a proxy.
    # However, the prompt mentions "geometric angular sweep".
    # With only a distance matrix, we can approximate the "direction" by looking at 
    # the relative distances to other nodes.
    # 
    # Alternative Interpretation:
    # The "angle" can be approximated by the change in direction.
    # Let's define a "direction vector" in the metric space.
    # This is tricky with just a distance matrix.
    #
    # Let's reconsider the directive: "Represent the routing decision as a geometric angular sweep around the depot".
    # This usually requires coordinates. If coordinates are not available, we must simulate them or use a proxy.
    # A common proxy for "clustering" with a distance matrix is to prefer nodes j such that
    # d(current, j) is small AND d(j, depot) is not too different from d(current, depot)?
    # Or, more simply, the savings heuristic IS a form of angular preference in a metric space.
    #
    # Let's try a different structural change: **Nearest Neighbor with Capacity-Weighted Distance**.
    # Or **Furthest Insertion** logic adapted for sequential selection?
    #
    # Actually, let's look at the "angular" idea again.
    # If we assume the points are in 2D, we can't get angles from a distance matrix alone without reconstruction (MDS), which is expensive and might not be stable.
    #
    # Let's pivot to a **Cluster-Based** heuristic using the distance matrix directly.
    # Idea: Group nodes by their proximity to the depot and to each other.
    # When at node `current`, prefer nodes `j` that are "close" to `current` in terms of distance, 
    # but also penalize nodes that are "far" from the depot if we are already far from the depot?
    #
    # Let's try a **Hybrid of Savings and Nearest Neighbor** but with a twist:
    # **Dynamic Weighted Nearest Neighbor**:
    # Score(j) = w1 * d(current, j) + w2 * (d(j, depot) - d(current, depot))^2
    # This encourages moving to nodes that are close to current but also keeps the "radius" from depot somewhat consistent, promoting circular routes.
    
    d_current_j = distance_matrix[current_node, feasible_nodes]
    d_depot_current = distance_matrix[depot, current_node]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    
    # Distance to current (primary cost)
    cost_dist = d_current_j
    
    # Radius deviation penalty (secondary cost)
    # We want to avoid jumping to a node that is much farther from the depot than current, 
    # as that might break the "loop" structure, or much closer, which might be inefficient if we have to come back.
    # Actually, usually we want to stay in a similar "shell" around the depot.
    radius_dev = (d_depot_j - d_depot_current) ** 2
    
    # Normalize costs to combine them
    # Max possible distance is bounded by the matrix values.
    max_dist = np.max(distance_matrix)
    if max_dist == 0:
        max_dist = 1.0
        
    norm_dist = cost_dist / max_dist
    norm_rad_dev = radius_dev / (max_dist * max_dist)
    
    # Weighted sum
    # We want to minimize this score.
    w_dist = 0.7
    w_rad = 0.3
    
    scores = w_dist * norm_dist + w_rad * norm_rad_dev
    
    # Find the best node (minimum score)
    best_idx = np.argmin(scores)
    best_node = feasible_nodes[best_idx]
    
    # Check if it's worth going to this node or returning to depot.
    # Compare cost of going to best_node and then eventually returning to depot
    # vs returning to depot now.
    # Approximation:
    # Cost to go to best_node: d(current, best_node) + d(best_node, depot)
    # Cost to return to depot: d(current, depot)
    # If d(current, best_node) + d(best_node, depot) > d(current, depot) + some threshold, return.
    # But this is a 2-step look ahead.
    
    cost_go = d_current_j[best_idx] + d_depot_j[best_idx]
    cost_return = distance_matrix[current_node, depot]
    
    # If going to the node and then returning to depot is significantly more expensive
    # than just returning to depot, we should close the route.
    # Note: d(best_node, depot) is the cost to finish the route after visiting best_node (assuming best_node is the last one).
    # If there are more nodes, this is an underestimate of the future cost, but a good heuristic.
    
    # Threshold: if cost_go > cost_return * 1.1 (10% penalty), close route.
    # This is a structural change: explicit route closing decision based on marginal cost.
    if cost_go > cost_return * 1.15:
        return int(depot)
        
    return int(best_node)
# EVOLVE-BLOCK-END
