"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 115.0  # Safety margin from 120s limit


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
    
    # Nearest-neighbor gap metric: estimate incremental cost of inserting each
    # candidate into the local neighborhood context.
    #
    # For each candidate v, compute:
    #   gap_v = dist(current, v) + dist(v, nearest_unvisited(v)) - dist(current, nearest_unvisited(current))
    #
    # This measures how much "extra" path length is introduced by choosing v
    # compared to the baseline of going to the globally nearest unvisited node
    # from current. Candidates that are close to both current and their own
    # nearest neighbor have low gaps (good local density fit).
    
    d_from_current = distance_matrix[current_node, unvisited_nodes]  # shape (n,)
    
    # Distance from each candidate to all other unvisited nodes
    d_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]  # shape (n, n)
    
    # For each candidate v (row i), find distance to its nearest OTHER unvisited node
    # Set diagonal to inf so a node doesn't pick itself
    np.fill_diagonal(d_unvisited, np.inf)
    nearest_unvisited_dist = np.min(d_unvisited, axis=1)  # shape (n,)
    
    # Baseline: distance from current to the nearest unvisited node
    baseline = np.min(d_from_current)
    
    # Gap score: how much extra cost does picking v add locally?
    # If v is the nearest to current, gap = 0 (best possible immediate step)
    # If v is far from current, gap is large
    # Additionally, if v's nearest neighbor is far, it suggests v is isolated
    # which will create a long edge later
    gap_scores = d_from_current + 0.3 * nearest_unvisited_dist - baseline
    
    # Also incorporate a mild destination lookahead for tie-breaking quality
    d_to_destination = distance_matrix[unvisited_nodes, destination_node]
    scores = gap_scores + 0.1 * d_to_destination
    
    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
