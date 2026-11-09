"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def _get_time_remaining():
    """Return seconds remaining before the 150s limit (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a combined heuristic that considers:
    1. Distance from current node
    2. Distance to the destination node (to avoid long closing edge)
    3. Isolation penalty (prefer nodes close to other unvisited nodes)
    
    Returns the index of the selected node.
    """
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 0.5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    candidates = unvisited_nodes
    n = len(candidates)

    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(candidates[0])

    if n == 2:
        # Pick the one that gives shorter total edge to destination
        d0 = distance_matrix[candidates[0], destination_node]
        d1 = distance_matrix[candidates[1], destination_node]
        if d0 < d1:
            return int(candidates[0])
        else:
            return int(candidates[1])

    # Score each candidate using multiple factors
    dist_from_current = distance_matrix[current_node, candidates]
    dist_to_dest = distance_matrix[candidates, destination_node]
    
    # Isolation penalty: for each candidate, compute the average distance to other unvisited nodes
    # We want to prefer nodes that are close to other unvisited nodes (not isolated)
    # This helps avoid creating long edges in the final tour
    
    # Compute pairwise distances among candidates
    # candidates is an array of node indices
    cand_indices = candidates.astype(int)
    
    # Get the submatrix of distances between all pairs of candidates
    # This is n x n where n = len(candidates)
    if n <= 50:  # Only do this for reasonable sizes to keep it fast
        cand_dist_matrix = distance_matrix[np.ix_(cand_indices, cand_indices)]
        # For each candidate, compute the mean distance to other unvisited nodes
        # Exclude self-distance (diagonal)
        np.fill_diagonal(cand_dist_matrix, 0)
        mean_dist_to_others = np.mean(cand_dist_matrix, axis=1)
    else:
        # For large n, use a simpler approximation
        mean_dist_to_others = np.zeros(n)

    # Adaptive alpha: when few nodes remain, destination matters more
    alpha = min(1.0, max(0.2, n / 15.0))
    
    # Isolation weight: penalize isolated nodes
    # The mean_dist_to_others should be small for well-connected nodes
    # We add a penalty proportional to how isolated a node is
    beta = 0.3  # weight for isolation penalty
    
    scores = dist_from_current + alpha * dist_to_dest + beta * mean_dist_to_others

    idx = np.argmin(scores)
    return int(candidates[idx])
# EVOLVE-BLOCK-END
