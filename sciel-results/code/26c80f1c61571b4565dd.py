"""Improved TSP constructive heuristic with look-ahead and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for time management
_start_time = None
_alpha = 0.4  # Weight for nearest neighbor vs. insertion cost (lower = more insertion focus)
_regret_weight = 0.15  # Weight for 2-step regret penalty


def _get_time_remaining():
    """Return seconds remaining before the 150s limit (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 140.0 - elapsed)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in the TSP tour.
    
    Uses a combination of:
    1. Nearest neighbor distance from current node
    2. Insertion cost: extra distance from routing through candidate before destination
    3. Regret penalty: how expensive the next step would be after visiting candidate
    
    This is a 2-step look-ahead heuristic that considers local proximity,
    tour-closing impact, and future connectivity.
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Safety check: if we're running out of time, fall back to nearest neighbor
    if _get_time_remaining() < 0.1:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Core distances
    dist_current = distance_matrix[current_node, unvisited_nodes]  # (n_unvisited,)
    dist_dest = distance_matrix[destination_node, unvisited_nodes]  # (n_unvisited,)
    dist_current_dest = distance_matrix[current_node, destination_node]
    
    # Insertion cost: extra distance if we go current->candidate->destination vs current->destination
    insertion_cost = dist_current + dist_dest - dist_current_dest
    
    # Regret: for each candidate i, what's the min distance from candidate_i to any OTHER unvisited node?
    # If this is large, picking candidate_i leaves us with an expensive next step.
    # We compute the min distance from each candidate to all other unvisited nodes (excluding itself).
    # For efficiency with n<=50, we can do this with a small loop or matrix ops.
    # dist matrix among unvisited: distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    unvisited_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    # Set diagonal to inf so we don't pick the node itself
    np.fill_diagonal(unvisited_dist, np.inf)
    # Min distance from each unvisited node to any other unvisited node
    min_next_dist = np.min(unvisited_dist, axis=1)  # (n_unvisited,)
    
    # Normalize components to [0, 1] using min-max
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / rng
    
    norm_dist = normalize(dist_current)
    norm_insert = normalize(insertion_cost)
    norm_regret = normalize(min_next_dist)
    
    # Combined score: lower is better
    score = _alpha * norm_dist + (1 - _alpha) * norm_insert + _regret_weight * norm_regret
    
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
