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
    
    # 2-step lookahead heuristic:
    # For each candidate c, estimate the cost of:
    #   1. Moving from current to c: d(current, c)
    #   2. Then from c to the best next node after c (greedy nearest among remaining unvisited,
    #      or to destination if c would be the last unvisited node)
    # This simulates the actual greedy trajectory better than abstract distance weighting.
    
    dist_from_current = distance_matrix[current_node, candidates]
    
    # For each candidate, compute the lookahead cost
    lookahead_costs = np.empty(n)
    for i in range(n):
        ci = candidates[i]
        if n == 2:
            # If only 2 nodes left, after picking ci we go to the other one, then destination
            other = candidates[1 - i]
            lookahead_costs[i] = dist_from_current[i] + distance_matrix[ci, other] + distance_matrix[other, destination_node]
        else:
            # After picking ci, the remaining candidates are all except ci
            remaining_mask = np.ones(n, dtype=bool)
            remaining_mask[i] = False
            remaining = candidates[remaining_mask]
            
            # Nearest neighbor from ci among remaining
            dists_from_ci = distance_matrix[ci, remaining]
            nn_idx = np.argmin(dists_from_ci)
            nn_node = remaining[nn_idx]
            nn_dist = dists_from_ci[nn_idx]
            
            # Total lookahead: current->ci + ci->nn
            lookahead_costs[i] = dist_from_current[i] + nn_dist
    
    # Combine with a small penalty for isolation to break ties and improve global structure
    # Compute min distance from each candidate to other unvisited nodes
    min_dist_to_other = np.empty(n)
    for i in range(n):
        ci = candidates[i]
        dists = distance_matrix[ci, candidates]
        dists[i] = np.inf
        min_dist_to_other[i] = np.min(dists)
    
    # Normalize isolation metric
    max_min_dist = np.max(min_dist_to_other)
    if max_min_dist > 0:
        isolation_norm = min_dist_to_other / max_min_dist
    else:
        isolation_norm = np.zeros(n)
    
    # Weight isolation penalty based on tour progress
    if n > 20:
        penalty_weight = 0.1
    elif n > 10:
        penalty_weight = 0.2
    else:
        penalty_weight = 0.3
    
    # Normalize lookahead costs for consistent scaling
    max_lookahead = np.max(lookahead_costs)
    if max_lookahead > 0:
        lookahead_norm = lookahead_costs / max_lookahead
    else:
        lookahead_norm = np.zeros(n)
    
    # Final score: primarily lookahead, with small isolation penalty
    score = lookahead_norm + penalty_weight * isolation_norm
    
    best_idx = np.argmin(score)
    return int(candidates[best_idx])
# EVOLVE-BLOCK-END
