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
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 0.5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    candidates = unvisited_nodes
    n = len(candidates)

    if n == 0:
        return destination_node
    if n == 1:
        return int(candidates[0])

    # Distance from current node to all candidates
    dist_from_current = distance_matrix[current_node, candidates]

    # Distance from each candidate to destination
    dist_to_dest = distance_matrix[candidates, destination_node]

    # 2-step lookahead: for each candidate i, find the minimum distance from i to any other candidate
    # This helps avoid creating long edges by preferring nodes that have close neighbors
    # Vectorized computation of min distance from each candidate to others (excluding self)
    # Use a large value for diagonal to ignore self-distance
    cand_dist_matrix = distance_matrix[np.ix_(candidates, candidates)]
    np.fill_diagonal(cand_dist_matrix, np.inf)
    min_next_dist = np.min(cand_dist_matrix, axis=1)

    # Isolation penalty: nodes that are far from other unvisited nodes get penalized
    # This encourages visiting clustered nodes together
    isolation_penalty = min_next_dist

    # Combined score: weighted sum of distance from current, distance to destination, and isolation
    # The destination bias helps close the tour; isolation penalty reduces long edges
    # Tuned weights based on TSP heuristics literature
    weight_current = 1.0
    weight_dest = 0.3
    weight_isolation = 0.2

    scores = (weight_current * dist_from_current + 
              weight_dest * dist_to_dest + 
              weight_isolation * isolation_penalty)

    return int(candidates[np.argmin(scores)])
# EVOLVE-BLOCK-END
