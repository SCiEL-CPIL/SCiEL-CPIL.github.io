"""CVRP Heuristic: Angular sweep with capacity-aware clustering."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using an angular sweep heuristic.
    
    Strategy:
    - Compute the prevailing direction of the current route (from depot to current node).
    - For each feasible candidate, compute the angular deviation from this direction.
    - Score = -angular_deviation * weight_angle - distance * weight_dist
    - This encourages continuing in the same angular sector, creating spatially clustered routes.
    - If at depot, pick the nearest feasible node to establish initial direction.
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
    n_feasible = len(feasible_nodes)
    
    # If at depot, pick the nearest feasible node to start the route
    if current_node == depot:
        dists = distance_matrix[current_node, feasible_nodes]
        min_idx = np.argmin(dists)
        return int(feasible_nodes[min_idx])
    
    # Extract coordinates from distance matrix (assuming Euclidean)
    # We need to infer 2D coordinates from the distance matrix
    # For a distance matrix D, we can use multidimensional scaling or assume
    # that the matrix is derived from 2D points.
    
    # Alternative approach: use the distance matrix directly to compute angles
    # relative to the depot-current vector.
    
    # Get distances from depot to all feasible nodes and from current to all feasible nodes
    d_depot_feasible = distance_matrix[depot, feasible_nodes]
    d_current_feasible = distance_matrix[current_node, feasible_nodes]
    d_current_depot = distance_matrix[current_node, depot]
    
    # Compute the "prevailing direction" using the law of cosines
    # For each candidate j, compute the angle at current_node between:
    # - vector from current to depot
    # - vector from current to j
    
    # Using law of cosines: cos(angle) = (a^2 + b^2 - c^2) / (2*a*b)
    # where a = d(current, depot), b = d(current, j), c = d(depot, j)
    
    # Avoid division by zero
    a = d_current_depot
    b = d_current_feasible
    c = d_depot_feasible
    
    # cos_theta = (a^2 + b^2 - c^2) / (2*a*b)
    numerator = a**2 + b**2 - c**2
    denominator = 2 * a * b
    
    # Handle case where denominator is zero
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = np.where(denominator > 1e-10, numerator / denominator, 0.0)
    
    # Clip to [-1, 1] for numerical stability
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Angular deviation: 0 means continuing in the same direction, pi means reversing
    # We want to minimize angular deviation, so we use (1 - cos_theta) as a proxy
    # Or better: use the actual angle in radians
    theta = np.arccos(cos_theta)  # 0 to pi
    
    # The "prevailing direction" is the direction from current back toward depot
    # We want candidates that continue in a similar angular sector
    # Small theta means the candidate is in roughly the same direction as depot from current
    
    # Actually, for angular sweep, we want to continue moving in the same general direction
    # away from the depot, creating a fan-like route pattern.
    
    # Let's reconsider: the angle at current_node between (current->depot) and (current->j)
    # If theta is small, j is in the same direction as depot from current (going back)
    # If theta is large (~pi), j is in the opposite direction (continuing away)
    
    # For a sweep pattern, we want to continue in the direction away from depot
    # So we want LARGE theta (close to pi) to keep going in the same general direction
    
    # Let's use a different formulation: compute the angle at the depot
    # between (depot->current) and (depot->j)
    
    a2 = d_current_depot  # depot to current
    b2 = d_depot_feasible  # depot to feasible
    c2 = d_current_feasible  # current to feasible
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta_depot = np.where((2 * a2 * b2) > 1e-10, 
                                   (a2**2 + b2**2 - c2**2) / (2 * a2 * b2), 
                                   1.0)
    
    cos_theta_depot = np.clip(cos_theta_depot, -1.0, 1.0)
    theta_depot = np.arccos(cos_theta_depot)  # 0 to pi
    
    # Small theta_depot means j is in the same angular sector as current from depot
    # This is what we want for clustering!
    
    # Score: prefer small angular deviation and short distance
    # weight_angle controls how much we prioritize angular clustering
    weight_angle = 1.5
    weight_dist = 0.5
    
    # Normalize distance to [0, 1] for balanced scoring
    max_dist = np.max(d_current_feasible) if np.max(d_current_feasible) > 0 else 1.0
    norm_dist = d_current_feasible / max_dist
    
    # Angular score: prefer small angles (close to current direction from depot)
    angle_score = theta_depot / np.pi  # 0 to 1, lower is better
    
    # Combined score: lower is better
    score = weight_angle * angle_score + weight_dist * norm_dist
    
    # Also consider capacity pressure: if rest_capacity is low, prefer nodes with lower demand
    # to leave room for more nodes in the sector
    demands_feasible = demands[feasible_nodes]
    max_demand = np.max(demands_feasible) if np.max(demands_feasible) > 0 else 1.0
    norm_demand = demands_feasible / max_demand
    
    # Capacity pressure: as rest_capacity decreases, penalize high-demand nodes more
    capacity_pressure = 1.0 - (rest_capacity / 50.0)  # 0 to 1
    demand_penalty = capacity_pressure * 0.5 * norm_demand
    
    total_score = score + demand_penalty
    
    # Check if continuing is worthwhile: if the best score is too high (bad), close route
    # Threshold: if even the best candidate has a high score, it's better to return to depot
    # This threshold depends on the problem scale; use a heuristic threshold
    if np.min(total_score) > 1.2:
        return int(depot)
    
    min_idx = np.argmin(total_score)
    
    return int(feasible_nodes[min_idx])
# EVOLVE-BLOCK-END
