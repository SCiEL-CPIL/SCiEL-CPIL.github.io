"""Improved constructive TSP heuristic: return-aware nearest neighbour with adaptive weighting and 2-step lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 120s limit


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in a TSP tour using a return-aware greedy heuristic with adaptive weighting
    and a 2-step lookahead to reduce myopia.
    
    The scoring combines:
    1. Forward distance (current -> candidate)
    2. Return distance (candidate -> destination) with adaptive weight
    3. Lookahead: minimum distance from candidate to any remaining unvisited node
    
    The weight on return distance increases as the tour nears completion,
    helping to close the loop effectively.
    """
    # Check deadline
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: nearest neighbor
        unv = np.asarray(unvisited_nodes)
        if unv.size == 0:
            return int(destination_node)
        if unv.size == 1:
            return int(unv[0])
        return int(unv[np.argmin(distance_matrix[current_node, unv])])
    
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    if n == 2:
        # For two remaining nodes, pick the one that minimizes the total path
        # current -> a -> b -> destination vs current -> b -> a -> destination
        a, b = int(unv[0]), int(unv[1])
        dm = distance_matrix
        cost_ab = dm[current_node, a] + dm[a, b] + dm[b, destination_node]
        cost_ba = dm[current_node, b] + dm[b, a] + dm[a, destination_node]
        return a if cost_ab <= cost_ba else b
    
    dm = distance_matrix
    
    # Forward cost: distance from current to each candidate
    d_forward = dm[current_node, unv]
    # Return cost: distance from each candidate to destination
    d_return = dm[unv, destination_node]
    
    # Adaptive weight: as fewer nodes remain, weight the return more heavily
    n_cities = distance_matrix.shape[0]
    frac_completed = 1.0 - (n / n_cities)
    
    # Smooth transition: w_return goes from ~0.3 to ~1.0
    w_return = 0.3 + 0.7 * frac_completed
    w_return = np.clip(w_return, 0.3, 1.0)
    
    # 2-step lookahead: for each candidate, estimate the cost of the next step
    # by finding the minimum distance from that candidate to any other unvisited node
    # This is O(n^2) but n is at most 50, so it's fast
    if n > 2:
        # min_next[i] = min distance from unv[i] to any other unvisited node (excluding destination)
        # We compute this efficiently using matrix operations
        # dist_from_unv[i, j] = dm[unv[i], unv[j]]
        dist_unv = dm[np.ix_(unv, unv)]
        # Set diagonal to infinity to exclude self
        np.fill_diagonal(dist_unv, np.inf)
        min_next = np.min(dist_unv, axis=1)
        
        # Lookahead weight: smaller when more nodes remain (less impact of next step)
        # and larger when fewer nodes remain
        w_lookahead = 0.2 + 0.3 * frac_completed
        w_lookahead = np.clip(w_lookahead, 0.2, 0.5)
    else:
        min_next = np.zeros(n)
        w_lookahead = 0.0
    
    # Total score: forward + w_return * return + w_lookahead * min_next
    score = d_forward + w_return * d_return + w_lookahead * min_next
    
    best_idx = np.argmin(score)
    return int(unv[best_idx])
# EVOLVE-BLOCK-END
