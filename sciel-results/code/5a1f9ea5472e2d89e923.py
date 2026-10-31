"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 110.0  # Leave 10s margin for 120s limit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Check time limit
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to nearest neighbor to ensure termination
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    if len(unvisited_nodes) == 0:
        return -1  # Should not happen if logic is correct, but safety

    # Get distances from current_node to all unvisited nodes
    dists_to_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from all unvisited nodes to the destination_node
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Heuristic: Minimize (dist(current, candidate) + dist(candidate, destination))
    # This balances the immediate step with the cost of returning to the start/destination.
    # We can tune the weight. 1.0 is a standard choice.
    scores = dists_to_current + dists_to_dest
    
    # Pick the unvisited node with the minimum score
    best_idx_in_unvisited = np.argmin(scores)
    best_node = unvisited_nodes[best_idx_in_unvisited]
    
    return int(best_node)
# EVOLVE-BLOCK-END
