"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE_MARGIN = 5.0  # seconds before the 150s limit to stop searching
_start_time = None


def _check_deadline():
    """Return True if we should stop searching (deadline approaching)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > (150.0 - _DEADLINE_MARGIN)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a multi-factor lookahead heuristic.
    Combines:
      1. Immediate distance from current node
      2. Distance from candidate to destination (final leg cost)
      3. Minimum distance from candidate to any other unvisited node (connectivity)
    Uses smooth weighting that adapts based on how many nodes remain.
    """
    n = len(unvisited_nodes)
    if n == 0:
        return destination_node
    if n == 1:
        return int(unvisited_nodes[0])
    if n == 2:
        # Choose the candidate that minimizes dist(current,c) + dist(c,dest)
        cands = unvisited_nodes
        d1 = distance_matrix[current_node, cands[0]] + distance_matrix[cands[0], destination_node]
        d2 = distance_matrix[current_node, cands[1]] + distance_matrix[cands[1], destination_node]
        return int(cands[0] if d1 <= d2 else cands[1])

    # If we're close to deadline, fall back to nearest neighbour for speed
    if _check_deadline():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Compute component distances
    dists_current_to_cand = distance_matrix[current_node, unvisited_nodes]  # (n,)
    dists_cand_to_dest = distance_matrix[unvisited_nodes, destination_node]  # (n,)
    
    # Minimum distance from each candidate to any other unvisited node (excluding itself)
    # This is a proxy for "how well-connected" the candidate is to the remaining set
    # For efficiency, compute row-wise min over the submatrix, then mask out self
    cand_submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]  # (n, n)
    # Set diagonal to inf so we don't pick self
    np.fill_diagonal(cand_submatrix, np.inf)
    min_dist_to_other = np.min(cand_submatrix, axis=1)  # (n,)

    # Adaptive weights based on remaining nodes
    # Early: emphasize immediate distance and connectivity
    # Late: emphasize destination distance
    frac_remaining = n / 50.0  # assuming ~50 city instances
    
    # Weight for immediate distance
    w_immediate = 1.0
    # Weight for destination distance: increases as we get closer to end
    w_dest = 0.3 + 1.7 * (1.0 - frac_remaining)
    # Weight for connectivity: moderate, decreases slightly as fewer nodes remain
    w_connect = 0.2 + 0.3 * frac_remaining

    # Normalize connectivity term to be comparable to distance terms
    # Use mean distance as a scale reference
    scale = np.mean(dists_current_to_cand) if np.mean(dists_current_to_cand) > 1e-10 else 1.0
    min_dist_normalized = min_dist_to_other / scale

    scores = (w_immediate * dists_current_to_cand +
              w_dest * dists_cand_to_dest +
              w_connect * min_dist_normalized)

    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
