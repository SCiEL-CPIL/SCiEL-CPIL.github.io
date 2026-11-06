"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline checking
_START_TIME = time.time()
_DEADLINE = 145.0  # 5 seconds margin before 150s limit

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: Minimize d(current, j) + d(j, depot) for feasible j.
    This favors nodes that are close to the current location and also close to the depot,
    creating compact routes and avoiding long detours that force expensive returns.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback: return depot to close route safely
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    # Convert to numpy array for fast operations
    unvisited = np.array(unvisited_nodes)
    
    # Filter for feasible nodes based on capacity
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return 0

    # Calculate distance from current node to all feasible nodes
    dist_current = distance_matrix[current_node, feasible_nodes]
    
    # Calculate distance from all feasible nodes to depot
    dist_depot = distance_matrix[feasible_nodes, depot]
    
    # Score: Sum of distances (Current->Node + Node->Depot)
    # This minimizes the "loop" cost, encouraging compact routes.
    scores = dist_current + dist_depot
    
    # Pick the node with the minimum score
    best_idx = np.argmin(scores)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
