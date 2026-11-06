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
    """
    Select the next node for a TSP tour using a lookahead-based heuristic.
    
    The heuristic considers:
    1. Immediate distance from current_node to candidate
    2. Estimated cost of connecting candidate to the next-best unvisited node
       (2-step lookahead)
    3. Proximity to destination_node as a tiebreaker/regularizer
    
    This is a structural change from pure nearest-neighbor to a local
    density-aware scoring that estimates the incremental tour cost.
    """
    _ensure_start_time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        # Exact 2-step lookahead: pick the candidate that minimizes
        # dist(current, cand1) + dist(cand1, cand2) + dist(cand2, dest)
        # vs dist(current, cand2) + dist(cand2, cand1) + dist(cand1, dest)
        c1 = int(unvisited_nodes[0])
        c2 = int(unvisited_nodes[1])
        cost1 = (distance_matrix[current_node, c1] + 
                 distance_matrix[c1, c2] + 
                 distance_matrix[c2, destination_node])
        cost2 = (distance_matrix[current_node, c2] + 
                 distance_matrix[c2, c1] + 
                 distance_matrix[c1, destination_node])
        return c1 if cost1 <= cost2 else c2
    
    # Check time budget early for large instances
    if _time_exceeded():
        # Fallback to nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # For larger sets, use a vectorized lookahead heuristic
    # Score each candidate u as:
    #   dist(current, u) + alpha * min_{v != u in unvisited} dist(u, v) + beta * dist(u, destination)
    #
    # The second term estimates the cost of leaving the local cluster.
    # We compute this efficiently using vectorized operations.
    
    d_from_current = distance_matrix[current_node, unvisited_nodes]  # shape (n,)
    
    # Compute the minimum distance from each unvisited node to any other unvisited node
    # This is the "nearest-neighbor gap" in the unvisited set
    # dist_matrix among unvisited nodes
    unvisited_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]  # shape (n, n)
    
    # Set diagonal to inf so we don't pick self
    np.fill_diagonal(unvisited_dist, np.inf)
    
    # Min distance from each node to its nearest other unvisited node
    nearest_other_dist = np.min(unvisited_dist, axis=1)  # shape (n,)
    
    # Distance to destination
    d_to_dest = distance_matrix[unvisited_nodes, destination_node]  # shape (n,)
    
    # Composite score: immediate cost + penalty for isolation + slight destination bias
    # alpha=0.5 balances the lookahead, beta=0.1 is a gentle pull toward destination
    alpha = 0.5
    beta = 0.1
    
    scores = d_from_current + alpha * nearest_other_dist + beta * d_to_dest
    
    best_idx = int(np.argmin(scores))
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
