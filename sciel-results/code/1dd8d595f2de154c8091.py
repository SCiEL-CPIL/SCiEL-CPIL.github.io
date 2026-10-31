"""Rewritten constructive TSP heuristic: look-ahead greedy with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_DEADLINE = None
_START_TIME = None


def _init_deadline():
    global _DEADLINE, _START_TIME
    if _DEADLINE is None:
        _START_TIME = time.perf_counter()
        # Leave a safety margin of ~5s before the 150s kill
        _DEADLINE = _START_TIME + 145.0


def _time_left():
    if _DEADLINE is None:
        return 145.0
    return _DEADLINE - time.perf_counter()


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city for a traveling-salesman tour.
    
    Uses a 2-step look-ahead greedy: for each candidate next node, estimate the
    cost of moving to it plus the cheapest edge from that node to any remaining
    unvisited node (or to destination if only one remains). This captures
    immediate + one-step-ahead cost, avoiding locally-cheap but structurally-bad
    choices.
    
    Falls back to pure nearest-neighbor if time budget is nearly exhausted.
    """
    _init_deadeline()
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        # Should not happen per contract, but guard
        return int(destination_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If very little time left, use fast nearest-neighbor
    if _time_left() < 0.1:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # 2-step look-ahead scoring
    # For each candidate u in unvisited_nodes:
    #   cost = d(current, u) + min_{v in unvisited \ {u}} d(u, v)
    # If only 1 remains after picking u, the "min to remaining" is d(u, destination)
    
    # Efficient vectorized computation:
    # dist_from_current: shape (n_unvisited,)
    dist_from_current = distance_matrix[current_node, unvisited_nodes]  # (n_u,)
    
    if n_unvisited == 2:
        # Special case: two nodes left. Evaluate both orderings.
        u0, u1 = unvisited_nodes[0], unvisited_nodes[1]
        # Option A: go to u0 then u1 then destination
        cost_a = dist_from_current[0] + distance_matrix[u0, u1] + distance_matrix[u1, destination_node]
        # Option B: go to u1 then u0 then destination
        cost_b = dist_from_current[1] + distance_matrix[u1, u0] + distance_matrix[u0, destination_node]
        if cost_a <= cost_b:
            return int(u0)
        else:
            return int(u1)
    
    # General case: n_unvisited >= 3
    # For each candidate u_i, we need min over v != u_i of d(u_i, v)
    # Build sub-matrix of distances among unvisited nodes
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]  # (n_u, n_u)
    
    # For each row i, find the minimum excluding the diagonal (self)
    # Set diagonal to inf, then argmin/min along axis=1
    sub_dist_no_diag = sub_dist.copy()
    np.fill_diagonal(sub_dist_no_diag, np.inf)
    min_next_cost = sub_dist_no_diag.min(axis=1)  # (n_u,)
    
    # Total score for each candidate
    scores = dist_from_current + min_next_cost
    
    # Pick the candidate with minimum score
    best_idx = int(np.argmin(scores))
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
