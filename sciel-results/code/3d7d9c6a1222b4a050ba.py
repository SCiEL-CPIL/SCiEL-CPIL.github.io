"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = None
_TIME_LIMIT = 130.0  # Safety margin before 150s kill

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _ensure_start_time()
    return _TIME_LIMIT - (time.time() - _START_TIME)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a degree-2 completion cost heuristic.
    
    For each candidate, we estimate the marginal cost of adding it to the partial tour.
    The partial tour is a path from start to current_node. When we add candidate,
    we extend the path. The "completion cost" considers how expensive it will be
    to eventually close the tour, focusing on the gap between the start side and
    the current end.
    
    Heuristic: For candidate c, the cost is:
      - dist(current, c) : immediate edge cost
      - Plus a penalty for how "isolated" c is relative to remaining nodes,
        approximated by the distance from c to the nearest node that is NOT
        adjacent to current (to avoid creating long gaps).
    
    Additionally, we consider the destination_node (final city) to bias toward
    paths that don't strand the destination far from the growing path.
    """
    _ensure_start_time()
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # If time is running low, fall back to simple nearest neighbor
    if _time_remaining() < 5.0:
        dists_from_cur = distance_matrix[current_node, unvisited_nodes]
        best_local_idx = np.argmin(dists_from_cur)
        return int(unvisited_nodes[best_local_idx])

    # Distances from current node to all unvisited nodes
    dists_from_cur = distance_matrix[current_node, unvisited_nodes]
    
    # Get the submatrix of distances between unvisited nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each candidate, compute the distance to the nearest OTHER unvisited node
    # (this approximates how "connected" the candidate is to the rest)
    np.fill_diagonal(sub_matrix, np.inf)
    min_dist_to_other = np.min(sub_matrix, axis=1)
    
    # Also consider distance to destination_node for each candidate
    # This helps ensure the destination isn't stranded
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # The degree-2 completion cost heuristic:
    # When we pick candidate c, we form edge (current, c).
    # The "gap" that must eventually be closed includes the start of the tour.
    # A good proxy: penalize candidates that are far from the cluster of remaining nodes,
    # because they create long edges that are hard to close.
    #
    # Score = w1 * dist(current, c) + w2 * min_dist_to_other[c] + w3 * dist(c, dest)
    #
    # We want to minimize immediate cost but also ensure the candidate is well-connected
    # to the remaining set (low min_dist_to_other) and not too far from destination.
    
    # Normalize components to be on similar scales
    max_dist_cur = np.max(dists_from_cur) if np.max(dists_from_cur) > 0 else 1.0
    max_dist_other = np.max(min_dist_to_other) if np.max(min_dist_to_other) > 0 else 1.0
    max_dist_dest = np.max(dists_to_dest) if np.max(dists_to_dest) > 0 else 1.0
    
    norm_cur = dists_from_cur / max_dist_cur
    norm_other = min_dist_to_other / max_dist_other
    norm_dest = dists_to_dest / max_dist_dest
    
    # Weights: emphasize immediate cost, but with meaningful penalties for
    # poor connectivity and destination distance
    w1 = 1.0   # immediate edge cost
    w2 = 0.5   # connectivity to remaining nodes
    w3 = 0.3   # distance to destination
    
    scores = w1 * norm_cur + w2 * norm_other + w3 * norm_dest
    
    # Select the candidate with the minimum score
    best_local_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_local_idx])
# EVOLVE-BLOCK-END
