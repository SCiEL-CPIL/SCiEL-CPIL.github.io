"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard (optional, but good practice per directive)
# The task is small (50 nodes), so this is mostly for safety.
START_TIME = time.time()
DEADLINE = START_TIME + 110.0  # 10s margin before 120s limit


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next customer to visit using a Clarke-Wright Savings-inspired heuristic.
    
    The heuristic minimizes: d(current, candidate) - d(depot, candidate)
    This is equivalent to maximizing the savings of linking current->candidate 
    instead of current->depot->candidate.
    """
    # Check deadline (though N=50 is fast)
    if time.time() > DEADLINE:
        return 0

    # Convert unvisited_nodes to a numpy array for vectorized operations
    unvisited = np.array(unvisited_nodes)
    
    if len(unvisited) == 0:
        return 0

    # Filter by capacity
    # demands is likely a list or array indexed by node id
    # We need to check if demands[node] <= rest_capacity
    feasible_mask = np.array([demands[node] <= rest_capacity for node in unvisited])
    
    if not np.any(feasible_mask):
        return 0

    feasible_nodes = unvisited[feasible_mask]

    # Calculate distances from current_node to all feasible nodes
    # distance_matrix is likely a 2D array
    d_current = distance_matrix[current_node, feasible_nodes]
    
    # Calculate distances from depot to all feasible nodes
    d_depot = distance_matrix[depot, feasible_nodes]

    # Score: Minimize d(current, node) - d(depot, node)
    # This promotes routes that move away from the depot in a consistent direction
    scores = d_current - d_depot

    # Find the index of the minimum score
    best_idx = np.argmin(scores)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
