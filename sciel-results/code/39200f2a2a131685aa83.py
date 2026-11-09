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
        return destination_node
    
    if n == 1:
        return int(candidates[0])
    
    # Structural change: use angle-based continuation heuristic
    # In Euclidean TSP, good tours tend to avoid sharp turns.
    # We approximate the "incoming direction" using the vector from destination to current
    # (since destination is where we ultimately need to go, this gives a sense of tour flow)
    # Then prefer candidates that continue in a similar direction.
    
    # Get coordinates implicitly via distance matrix - but we don't have coords.
    # Alternative structural approach: use a 2-opt lookahead with limited horizon.
    # For each candidate, estimate the additional cost of inserting it now vs. later.
    
    # Simpler structural change: use nearest neighbor but with a "lookahead" penalty.
    # For each candidate, compute the distance to its nearest unvisited neighbor (excluding itself).
    # Penalize candidates whose nearest neighbor is far away (they'd create isolated edges).
    
    dist_from_current = distance_matrix[current_node, candidates]
    
    # Compute for each candidate, the minimum distance to other unvisited nodes
    # This is O(n^2) but n <= 50, so it's fast
    min_dist_to_other = np.empty(n)
    for i in range(n):
        ci = candidates[i]
        # Distance from ci to all other candidates
        dists = distance_matrix[ci, candidates]
        dists[i] = np.inf  # Exclude self
        min_dist_to_other[i] = np.min(dists)
    
    # Normalize the min_dist_to_other to [0, 1] range for consistent scaling
    if np.max(min_dist_to_other) > 0:
        min_dist_norm = min_dist_to_other / np.max(min_dist_to_other)
    else:
        min_dist_norm = np.zeros(n)
    
    # Score: primarily nearest neighbor, but penalize candidates that are "isolated"
    # (far from all other unvisited nodes) because visiting them now creates long edges
    # that can't be easily fixed later.
    
    # Weight the isolation penalty based on tour progress
    if n > 20:
        # Early: less penalty, we're establishing the tour skeleton
        penalty_weight = 0.3
    elif n > 10:
        penalty_weight = 0.5
    else:
        # Late: higher penalty to avoid creating long final edges
        penalty_weight = 0.8
    
    score = dist_from_current + penalty_weight * min_dist_norm * np.max(dist_from_current)
    
    best_idx = np.argmin(score)
    return int(candidates[best_idx])
# EVOLVE-BLOCK-END
