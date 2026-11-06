"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache to avoid recomputing or converting types if the evaluator
# passes the same data structures repeatedly.
_cache = {}
_start_time = None

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time, _cache
    
    # Initialize time on first call
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 120s limit, stop at 110s to be safe.
    # If we are running out of time, we should just close the route immediately
    # or pick the nearest feasible node to finish quickly.
    elapsed = time.time() - _start_time
    if elapsed > 110.0:
        # Emergency exit: return depot to close route, or nearest feasible.
        # Returning depot is safer to ensure valid route closure.
        return depot

    # Handle empty unvisited list
    if not unvisited_nodes:
        return depot

    # Convert inputs to numpy arrays for speed if not already cached/compatible
    # We assume distance_matrix and demands are arrays or list-like.
    # We will use the passed arrays directly for this step to avoid cache invalidation issues
    # if the evaluator changes them (though unlikely).
    
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    demands_arr = np.array(demands, dtype=float)
    dist_mat = np.array(distance_matrix, dtype=float)

    # 1. Filter for capacity feasibility
    feasible_mask = demands_arr[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]

    if len(feasible_nodes) == 0:
        return depot

    # 2. Calculate scores
    # Heuristic: Minimize (distance from current to node) + (distance from node to depot)
    # This encourages visiting nodes that are close to the current position
    # and also close to the depot, effectively creating compact routes.
    
    dists_from_current = dist_mat[current_node, feasible_nodes]
    dists_to_depot = dist_mat[feasible_nodes, depot]
    
    scores = dists_from_current + dists_to_depot
    
    # 3. Select the node with the minimum score
    min_idx = np.argmin(scores)
    next_node = feasible_nodes[min_idx]
    
    return int(next_node)
# EVOLVE-BLOCK-END
