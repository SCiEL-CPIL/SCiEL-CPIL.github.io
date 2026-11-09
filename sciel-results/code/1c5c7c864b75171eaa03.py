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
    if _get_time_remaining() < 1.0:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Score each candidate by: dist(current, candidate) + dist(candidate, destination)
    # This lookahead helps avoid getting stuck far from the destination
    candidates = unvisited_nodes
    if len(candidates) == 0:
        return destination_node
    
    if len(candidates) == 1:
        return int(candidates[0])
    
    # Compute scores: distance from current to candidate + distance from candidate to destination
    dist_to_candidate = distance_matrix[current_node, candidates]
    dist_candidate_to_dest = distance_matrix[candidates, destination_node]
    scores = dist_to_candidate + dist_candidate_to_dest
    
    return int(candidates[np.argmin(scores)])
# EVOLVE-BLOCK-END
