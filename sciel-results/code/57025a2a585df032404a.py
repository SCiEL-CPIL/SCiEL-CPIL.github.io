"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for tracking and caching
_start_time = None
_route_cache = {}
_instance_id = 0
_deadline = 145.0  # 5s buffer before 150s limit


def _get_instance_key(depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """Create a key to identify the instance."""
    # Use the first few elements of the distance matrix and demands as a fingerprint
    return (tuple(demands[:5]), tuple(distance_matrix[0, :5]), depot)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time, _route_cache, _instance_id
    
    # Initialize deadline tracking on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: stop if we've exceeded deadline
    elapsed = time.time() - _start_time
    if elapsed > _deadline:
        return depot
    
    if len(unvisited_nodes) == 0:
        return depot
    
    # Check if all unvisited nodes are infeasible
    feasible_nodes = []
    for node in unvisited_nodes:
        if demands[node] <= rest_capacity:
            feasible_nodes.append(node)
    
    if len(feasible_nodes) == 0:
        return depot
    
    # Convert to numpy array for faster operations
    feasible_arr = np.array(feasible_nodes)
    
    # Compute distances from current node to all feasible nodes
    dists = distance_matrix[current_node, feasible_arr]
    
    # Compute distance from each feasible node back to depot
    dists_to_depot = distance_matrix[feasible_arr, depot]
    
    # Strategy: Use a combined score that considers:
    # 1. Distance from current node to candidate
    # 2. Distance from candidate to depot (to prefer nodes that keep route short)
    # 3. Remaining capacity after serving candidate (prefer using capacity efficiently)
    
    # Score: minimize dist_from_current + alpha * dist_to_depot
    # But also consider that we want to fill the route efficiently
    # Use a simple heuristic: nearest neighbor with a penalty for leaving too much capacity
    
    # Primary: nearest feasible node
    # Secondary: among close nodes, prefer those that leave less remaining capacity
    # (i.e., larger demand) to pack the route better
    
    # Compute a combined score
    # Normalize distances to make them comparable
    max_dist = max(np.max(dists), np.max(dists_to_depot), 1.0)
    
    # Weighted score: 0.7 * normalized_dist_from_current + 0.3 * normalized_dist_to_depot
    # But also add a small bonus for higher demand (to fill capacity)
    demands_feasible = demands[feasible_arr]
    max_demand = max(np.max(demands_feasible), 1.0)
    
    # Score: lower is better
    # dist_from_current dominates, but we add a small penalty for not using capacity well
    score = dists + 0.1 * dists_to_depot - 0.05 * (demands_feasible / max_demand) * np.max(dists)
    
    # Find the node with the minimum score
    best_idx = np.argmin(score)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
