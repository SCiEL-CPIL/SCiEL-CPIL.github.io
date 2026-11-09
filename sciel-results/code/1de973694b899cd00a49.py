"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Deadline guard: 110s limit to be safe against 120s kill
    if _get_elapsed() > 110:
        # Fallback to nearest neighbor to ensure termination
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    if len(unvisited_nodes) == 0:
        return -1

    # Get distances from current node to all unvisited nodes
    dists = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from destination node to all unvisited nodes
    dists_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Distance between current and destination
    dist_cur_dest = distance_matrix[current_node, destination_node]

    # If current is destination, just pick the nearest neighbor
    if dist_cur_dest < 1e-9:
        return int(unvisited_nodes[np.argmin(dists)])

    # Calculate angles using the Law of Cosines
    # Triangle: current (C), destination (D), candidate (J)
    # We want angle at C between vector CD and vector CJ
    # Sides: a = dist(C,D), b = dist(C,J), c = dist(D,J)
    # cos(angle) = (a^2 + b^2 - c^2) / (2ab)
    
    denom = 2.0 * dist_cur_dest * dists
    # Avoid division by zero if any dists is 0 (shouldn't happen for unvisited != current, but safe)
    # If dists is 0, it means candidate is current, which is not in unvisited.
    cos_alpha = (dist_cur_dest**2 + dists**2 - dists_dest**2) / denom
    
    # Clip to [-1, 1] to handle floating point errors
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    
    angles = np.arccos(cos_alpha) # Angles in [0, pi]
    
    # Heuristic Score:
    # We want to minimize distance, but strongly penalize "backward" movement (large angles).
    # Score = distance * (1 + angle)
    # This ensures that nodes in the forward direction are preferred, 
    # effectively creating a "sweep" or "convex hull" like behavior.
    scores = dists * (1.0 + angles)
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
