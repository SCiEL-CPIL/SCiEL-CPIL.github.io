"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 3.0


def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated VRP: pick next customer or depot (0) to close route.
    
    Strategy: Clarke-Wright Savings-based greedy construction.
    Instead of just minimizing immediate distance, we evaluate the 'savings' 
    of visiting a node now. The savings for visiting node v from current node u
    is roughly: d(depot, u) + d(depot, v) - d(u, v).
    However, since we are building a route sequentially, a more direct metric
    for the next step in a savings context is often just the distance itself,
    BUT the key insight of Savings is that we should prefer nodes that allow
    us to 'merge' two trips into one efficiently.
    
    In a sequential construction, the 'Savings' heuristic often degenerates to
    nearest neighbor if implemented naively. To make it distinct, we use a
    modified score that penalizes nodes that are far from the depot relative
    to their proximity to the current node, effectively encouraging routes
    that 'wrap around' the depot or stay in tight clusters, which is the
    spirit of savings.
    
    Actually, a more robust structural change is to use a **2-opt lookahead**
    or a **Savings-like penalty** on the return trip.
    
    Let's use a score: d(current, v) - beta * d(v, depot)
    This encourages going to nodes that are close to current BUT also close
    to depot (so we don't end up far from depot when we have to return).
    Wait, that's not quite savings.
    
    Let's stick to the core Savings idea adapted for sequential choice:
    The savings of adding v to the current route ending at u, compared to
    having u and v in separate routes (which would require d(u, depot) + d(depot, v))
    is: (d(u, depot) + d(depot, v)) - d(u, v).
    We want to MAXIMIZE this savings.
    Minimizing -savings is equivalent to minimizing d(u, v) - d(u, depot) - d(depot, v).
    Since d(u, depot) is constant for a fixed u, minimizing d(u, v) - d(depot, v)
    is the key term.
    
    So, Score(v) = d(current, v) - d(depot, v).
    We pick the feasible v that minimizes this score.
    This favors nodes that are close to current but *not* too far from depot
    relative to their distance from current.
    """
    if _time_exceeded():
        return 0

    unvisited = list(unvisited_nodes)
    if not unvisited:
        return 0

    # Filter by capacity
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    if not feasible:
        return 0

    feasible_arr = np.array(feasible)
    
    # Calculate savings-based score
    # Savings of merging v into current route vs separate trips:
    # S = d(current, depot) + d(depot, v) - d(current, v)
    # We want to maximize S.
    # Equivalent to minimizing: d(current, v) - d(depot, v) - d(current, depot)
    # Since d(current, depot) is constant, we minimize: d(current, v) - d(depot, v)
    
    dist_from_current = distance_matrix[current_node, feasible_arr]
    dist_from_depot = distance_matrix[depot, feasible_arr]
    
    # The savings score to minimize
    scores = dist_from_current - dist_from_depot
    
    # Pick the node with minimum score (max savings)
    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
