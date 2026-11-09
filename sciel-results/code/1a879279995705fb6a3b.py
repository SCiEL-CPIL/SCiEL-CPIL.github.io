"""Combined CVRP heuristic: savings-based selection with capacity-awareness and angular clustering."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = None
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a savings-based heuristic with capacity-awareness.
    Returns 0 (depot) if no feasible node can be added or if savings are negative.
    
    Strategy:
    - When at depot, pick the nearest feasible node to start the route.
    - Otherwise, compute savings for each feasible node:
      saving(j) = dist(current, depot) + dist(depot, j) - dist(current, j)
    - If max saving is negative, close the route (go to depot).
    - Otherwise, pick the node with maximum saving, with a small capacity penalty
      to encourage efficient use of remaining capacity.
    """
    global _start_time, _DEADLINE
    
    if _start_time is None:
        _start_time = time.time()
        _DEADLINE = _start_time + _TIMEOUT
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return int(depot)
    
    # Convert unvisited_nodes to numpy array if it's a list
    if isinstance(unvisited_nodes, (list, tuple)):
        unvisited_arr = np.array(unvisited_nodes, dtype=int)
    else:
        unvisited_arr = np.asarray(unvisited_nodes, dtype=int)
    
    # If no unvisited nodes, return depot
    if len(unvisited_arr) == 0:
        return int(depot)
    
    # Find feasible nodes: those whose demand fits in rest_capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    
    # If no feasible nodes, return depot
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited_arr[feasible_mask]
    
    # If at depot, pick the nearest feasible node to start the route
    if current_node == depot:
        dists = distance_matrix[current_node, feasible_nodes]
        min_idx = np.argmin(dists)
        return int(feasible_nodes[min_idx])
    
    n_feasible = len(feasible_nodes)
    
    # Compute distances
    d_current_depot = distance_matrix[current_node, depot]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    d_current_j = distance_matrix[current_node, feasible_nodes]
    
    # Calculate savings for each feasible node
    savings = d_current_depot + d_depot_j - d_current_j
    
    # If the maximum saving is negative, it is cheaper to close the route
    if np.max(savings) < 0:
        return int(depot)
    
    # Look-ahead: estimate the cost of the step *after* visiting candidate j.
    # For each candidate j, find the cheapest feasible next step (to another
    # unvisited node or back to depot), then subtract a fraction of that
    # continuation cost from the immediate savings. This reduces myopia.
    #
    # For speed, we approximate: for each j, the best next move is either
    #   (a) back to depot (cost d_depot_j), or
    #   (b) to the nearest other feasible node k != j.
    # We use a vectorized approach: compute pairwise distances among feasible
    # nodes, then for each j take min(d_depot_j, min_{k != j} d_jk).
    
    if n_feasible > 1:
        # Pairwise distance matrix among feasible nodes
        d_ff = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
        # For each row j, the min distance to any other feasible node
        # Set diagonal to inf so j doesn't pick itself
        np.fill_diagonal(d_ff, np.inf)
        min_next_other = np.min(d_ff, axis=1)
    else:
        min_next_other = np.array([0.0])
    
    # Continuation cost: min of going to depot or to nearest other feasible
    continuation = np.minimum(d_depot_j, min_next_other)
    
    # Look-ahead penalty: subtract a fraction of continuation cost
    # Weight 0.3 balances immediate savings with future cost
    lookahead_weight = 0.3
    adjusted_savings = savings - lookahead_weight * continuation
    
    # Capacity-aware refinement:
    # Penalize nodes that would leave very little capacity (risk of stranding)
    # or nodes with very small demand (wasting capacity)
    demand_ratios = demands[feasible_nodes] / rest_capacity
    
    # Smooth penalty: penalize deviation from ideal fill ratio (~0.5)
    # Use a wider penalty for over-filling (ratio > 0.7) since stranding
    # is more costly than under-filling
    overfill_penalty = np.maximum(demand_ratios - 0.7, 0.0) ** 2
    underfill_penalty = np.maximum(0.3 - demand_ratios, 0.0) ** 2
    capacity_penalty = 0.15 * (overfill_penalty + 0.5 * underfill_penalty)
    
    # Scale penalty by the distance to the candidate (larger distances
    # amplify the impact of poor capacity decisions)
    adjusted_savings = adjusted_savings - capacity_penalty * d_current_j
    
    # Angular clustering: encourage spatial coherence by penalizing nodes
    # that deviate significantly from the prevailing route direction.
    # Compute the angle from depot to each feasible node relative to the
    # angle from depot to current_node. Prefer nodes in the same angular sector.
    
    # Get coordinates implicitly from distance matrix? No, we only have distances.
    # Instead, use a proxy for angular deviation: the difference in "direction"
    # can be approximated by comparing distances to depot.
    # 
    # Alternative: use the law of cosines to compute angles.
    # For triangle (depot, current, j):
    #   d_depot_current, d_depot_j, d_current_j
    # Angle at depot between current and j:
    #   cos(theta) = (d_depot_current^2 + d_depot_j^2 - d_current_j^2) / (2 * d_depot_current * d_depot_j)
    # 
    # We want to minimize angular deviation from the "prevailing direction".
    # The prevailing direction is roughly the direction from depot to current_node.
    # Nodes that are in the same direction (small angle at depot) are preferred.
    
    d_depot_current = distance_matrix[depot, current_node]
    
    # Compute angles at depot between current_node and each feasible node
    if d_depot_current > 1e-10:
        # cos(theta) = (a^2 + b^2 - c^2) / (2ab) where a=d_depot_current, b=d_depot_j, c=d_current_j
        cos_theta = (d_depot_current**2 + d_depot_j**2 - d_current_j**2) / (2 * d_depot_current * d_depot_j)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)  # angle at depot, in radians [0, pi]
        
        # Penalize large angular deviations. Nodes with small theta (same direction)
        # are preferred. Use a smooth penalty that grows with theta.
        # Weight 0.05 is modest to not override savings entirely.
        angular_weight = 0.05
        angular_penalty = angular_weight * theta * d_current_j
    else:
        angular_penalty = np.zeros(n_feasible)
    
    adjusted_savings = adjusted_savings - angular_penalty
    
    # Pick the node with the maximum adjusted saving
    max_idx = np.argmax(adjusted_savings)
    
    return int(feasible_nodes[max_idx])
# EVOLVE-BLOCK-END
