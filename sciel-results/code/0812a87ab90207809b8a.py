"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 118.0  # Safety margin from 150s limit (evaluator kills at 150s)


def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _TIME_LIMIT


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    _ensure_start_time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if _time_exceeded():
        # Fallback to nearest neighbor if time is running out
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Structural change: use a local density / nearest-neighbor gap metric
    # instead of a simple weighted sum of entry and exit distances.
    # 
    # For each candidate node, we estimate the "local tour cost" by looking at
    # how well it fits into the local neighborhood context. Specifically:
    # 1. Entry cost: distance from current_node to candidate
    # 2. Exit cost: distance from candidate to destination_node
    # 3. Local density penalty: how isolated the candidate is among unvisited nodes
    #    (penalize candidates that are far from other unvisited nodes, as they
    #    will be harder to integrate later)
    #
    # The key insight: a good next node should be close to current (low entry),
    # not too far from destination (manageable exit), AND have other unvisited
    # nodes nearby so the remaining tour doesn't have to stretch far.
    
    d_from_current = distance_matrix[current_node, unvisited_nodes]
    d_to_destination = distance_matrix[unvisited_nodes, destination_node]
    
    # Compute local density: for each candidate, the mean distance to the
    # k-nearest unvisited nodes (excluding itself). This captures how "clustered"
    # the candidate is with respect to the remaining cities.
    # Using a small k (e.g., 3) keeps it local and fast.
    k = min(3, n_unvisited - 1)
    
    if k > 0:
        # distance_matrix[unvisited_nodes, unvisited_nodes] gives pairwise distances
        dm_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        # Set diagonal to 0 (self-distance)
        np.fill_diagonal(dm_unvisited, 0.0)
        # For each row, get the k smallest distances (including 0 for self)
        # We want the k smallest non-zero distances, so we sort and take [1:k+1]
        sorted_dists = np.sort(dm_unvisited, axis=1)
        # Mean of the k nearest neighbors (excluding self)
        local_density = np.mean(sorted_dists[:, 1:k+1], axis=1)
    else:
        local_density = np.zeros(n_unvisited)
    
    # Normalize local density to [0, 1] range for consistent weighting
    ld_min = np.min(local_density)
    ld_max = np.max(local_density)
    if ld_max > ld_min:
        local_density_norm = (local_density - ld_min) / (ld_max - ld_min)
    else:
        local_density_norm = np.zeros(n_unvisited)
    
    # Weighted combination:
    # - 50% weight on entry distance (immediate cost)
    # - 30% weight on exit distance (future cost to destination)
    # - 20% weight on local density penalty (harder to integrate isolated nodes)
    w_entry = 0.5
    w_exit = 0.3
    w_density = 0.2
    
    # Normalize entry and exit to comparable scales
    d_from_current_norm = d_from_current / (np.max(d_from_current) + 1e-10)
    d_to_destination_norm = d_to_destination / (np.max(d_to_destination) + 1e-10)
    
    scores = (w_entry * d_from_current_norm + 
              w_exit * d_to_destination_norm + 
              w_density * local_density_norm)
    
    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
