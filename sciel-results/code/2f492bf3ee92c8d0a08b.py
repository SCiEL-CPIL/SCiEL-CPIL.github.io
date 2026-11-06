"""TSP constructive heuristic: boundary-expansion (convex-hull-inspired) scoring,
implemented with cheap vectorized distance-matrix lookups (no per-step hull)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 115.0  # Safety margin from 120s wall-clock limit


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
    Pick the next city for a step-by-step TSP tour.

    Structural idea (boundary expansion / convex-hull containment):
    A short closed tour on Euclidean points tends to sweep the point cloud
    without enclosing large unvisited pockets. We approximate "how much a
    candidate expands the boundary" cheaply: a node that is far from the rest
    of the unvisited set would push the partial tour outward (long detour /
    long return), so we penalize it. A node well-connected to the remaining
    cluster is "interior" and is preferred.

    score(candidate) =
        dist(current, candidate)                      # immediate edge
      + w_boundary * min_{v != candidate} dist(candidate, v)   # boundary/detour penalty
      + w_dest     * dist(candidate, destination)               # end-game return

    Weights are kept close to pure nearest-neighbor (the strongest known
    baseline) so the geometric term acts as a light regularizer; the exact
    2-step endgame and a time guard preserve robustness.
    """
    _ensure_start_time()

    n_unvisited = len(unvisited_nodes)

    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Exact 2-step endgame: minimize the closed remainder of the tour.
    if n_unvisited == 2:
        i0 = int(unvisited_nodes[0])
        i1 = int(unvisited_nodes[1])
        cost0 = (distance_matrix[current_node, i0] +
                 distance_matrix[i0, i1] +
                 distance_matrix[i1, destination_node])
        cost1 = (distance_matrix[current_node, i1] +
                 distance_matrix[i1, i0] +
                 distance_matrix[i0, destination_node])
        return i0 if cost0 <= cost1 else i1

    # Time guard: fall back to fast nearest-neighbor if the budget is nearly gone.
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Immediate distance from current node to each candidate.
    dist_from_current = distance_matrix[current_node, unvisited_nodes]

    # Boundary/detour penalty: nearest unvisited neighbor of each candidate
    # (proxy for how far the candidate extends the partial-tour frontier).
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    np.fill_diagonal(sub_dist, np.inf)
    min_outgoing = np.min(sub_dist, axis=1)

    # End-game return cost to destination.
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]

    # Light regularizer weights, close to pure nearest-neighbor.
    if n_unvisited <= 5:
        w_boundary = 0.15
        w_dest = 0.30
    elif n_unvisited <= 10:
        w_boundary = 0.15
        w_dest = 0.10
    else:
        w_boundary = 0.10
        w_dest = 0.0

    score = dist_from_current + w_boundary * min_outgoing + w_dest * dist_to_dest

    idx = int(np.argmin(score))
    return int(unvisited_nodes[idx])
# EVOLVE-BLOCK-END
