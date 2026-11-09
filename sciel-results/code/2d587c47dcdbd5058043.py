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
    
    candidates = unvisited_nodes
    n = len(candidates)
    
    if n == 0:
        return destination_node
    
    if n == 1:
        return int(candidates[0])
    
    # Compute distances from current to all candidates
    dist_to_candidate = distance_matrix[current_node, candidates]
    
    # If there are few candidates, use a lookahead that considers returning to destination
    # This helps avoid getting stuck far from the destination near the end of the tour
    if n <= 10:
        dist_candidate_to_dest = distance_matrix[candidates, destination_node]
        # Weighted combination: emphasize distance from current, but also consider
        # the cost of going from candidate to destination
        # Use a blend that reduces as we get closer to the end
        weight = min(1.0, n / 10.0)
        scores = dist_to_candidate + weight * dist_candidate_to_dest
        return int(candidates[np.argmin(scores)])
    
    # For larger sets, use nearest neighbor with a slight bias toward destination
    # when we're in the later stages of the tour
    if n <= 20:
        dist_candidate_to_dest = distance_matrix[candidates, destination_node]
        weight = 0.5
        scores = dist_to_candidate + weight * dist_candidate_to_dest
        return int(candidates[np.argmin(scores)])
    
    # For early stages, just use nearest neighbor for speed and simplicity
    return int(candidates[np.argmin(dist_to_candidate)])
# EVOLVE-BLOCK-END
