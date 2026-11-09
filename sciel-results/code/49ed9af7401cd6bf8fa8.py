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
        return int(destination_node)

    if n == 1:
        return int(candidates[0])

    # Compute distance from current to each candidate
    dist_current = distance_matrix[current_node, candidates]

    # If there's a destination to go back to, incorporate look-ahead cost
    # Score = dist(current -> candidate) + alpha * dist(candidate -> destination)
    # This is a simple 2-step look-ahead that uses the destination_node parameter
    if destination_node is not None and destination_node >= 0:
        dist_dest = distance_matrix[candidates, destination_node]
        # Use a weighted combination: primarily nearest, but penalize candidates
        # that would be far from the eventual destination
        alpha = 0.3  # weight for the look-ahead term
        scores = dist_current + alpha * dist_dest
    else:
        scores = dist_current

    # Find the best candidate
    best_idx = int(np.argmin(scores))
    return int(candidates[best_idx])
# EVOLVE-BLOCK-END
