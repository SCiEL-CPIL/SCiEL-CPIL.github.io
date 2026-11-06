"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Simple global start time for deadline guard
_start_time = time.time()
_DEADLINE = 145.0  # 5s safety margin from 150s

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 1-step lookahead heuristic.
    Minimizes d(current, candidate) + d(candidate, nearest_neighbor_of_candidate).
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback to nearest neighbor if time is tight
        if len(unvisited_nodes) == 0:
            return -1
        if len(unvisited_nodes) == 1:
            return int(unvisited_nodes[0])
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # 1. Distance from current node to all unvisited candidates
    # Shape: (n_unvisited,)
    dists_from_current = distance_matrix[current_node, unvisited_nodes]

    # 2. Distance matrix between unvisited nodes
    # Shape: (n_unvisited, n_unvisited)
    # We need to find the nearest neighbor for each candidate j in unvisited_nodes
    # excluding j itself.
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Set diagonal to infinity to prevent selecting self as neighbor
    np.fill_diagonal(sub_matrix, np.inf)
    
    # 3. Find the nearest neighbor index for each row (candidate)
    # nearest_neighbor_indices[i] is the index in unvisited_nodes of the closest node to unvisited_nodes[i]
    nearest_neighbor_indices = np.argmin(sub_matrix, axis=1)
    
    # 4. Get the distance to that nearest neighbor
    # dists_to_next[i] is the distance from unvisited_nodes[i] to its nearest neighbor
    dists_to_next = sub_matrix[np.arange(n_unvisited), nearest_neighbor_indices]
    
    # 5. Total estimated cost: d(current, j) + d(j, nearest(j))
    total_costs = dists_from_current + dists_to_next
    
    # 6. Select the candidate with the minimum total cost
    best_idx = np.argmin(total_costs)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
