"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_DEADLINE_SECONDS = 115.0  # safety margin under 120s limit
_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _check_deadline():
    return _get_elapsed() > _DEADLINE_SECONDS


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a combined heuristic:
    - If we're running out of time, fall back to nearest neighbor.
    - Otherwise, use a "lookahead" or "MST-leaf-inspired" heuristic that considers
      both proximity to current node and future connection cost to destination.
    
    The strategy: score each candidate by a weighted combination of:
      1. Distance from current_node to candidate (local cost)
      2. Distance from candidate to destination_node (future cost, to avoid
         leaving destination as a distant final jump)
    
    This is a simple but effective refinement of nearest-neighbor that avoids
    the classic NN pitfall of ending up far from the start/destination.
    """
    # Deadline guard: if we're over budget, use fastest fallback
    if _check_deadline():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Get distances from current node to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from each unvisited to the destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Normalize to [0, 1] for combination
    max_dfc = np.max(dist_from_current)
    max_dtd = np.max(dist_to_dest)
    
    if max_dfc > 0:
        norm_dfc = dist_from_current / max_dfc
    else:
        norm_dfc = np.zeros_like(dist_from_current)
    
    if max_dtd > 0:
        norm_dtd = dist_to_dest / max_dtd
    else:
        norm_dtd = np.zeros_like(dist_to_dest)
    
    # Weighted score: primarily local cost, secondarily future cost
    # The weight on future cost increases as we visit more nodes (fewer remaining)
    # This is a simple adaptive weight
    total_nodes = distance_matrix.shape[0]
    fraction_remaining = n_unvisited / max(total_nodes, 1)
    
    # When few nodes remain, heavily weight the distance to destination
    # When many remain, weight local proximity more
    w_future = 1.0 - fraction_remaining  # 0 when many remain, 1 when few remain
    w_local = 1.0  # always include local cost
    
    score = w_local * norm_dfc + w_future * norm_dtd
    
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
