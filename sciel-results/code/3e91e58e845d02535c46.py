"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# ---- Deadline guard -------------------------------------------------------
# The evaluator kills the program at 150 s wall-clock. We keep an explicit
# safety margin so we can always return a valid result.
_DEADLINE = 145.0  # seconds from process start before we stop doing "smart" work
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _time_remaining():
    return max(0.0, _DEADLINE - _elapsed())


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # If we are running low on time, fall back to the cheapest valid choice
    # (pure nearest neighbour) so we never risk a timeout.
    if _time_remaining() < 1.0:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Vectorised scoring over all unvisited candidates.
    # Primary heuristic: insertion cost  d(current, c) + d(c, destination).
    # This is the classic "look-ahead / 2-point" constructive heuristic that
    # outperforms pure nearest neighbour on Euclidean TSP instances.
    d_cur = distance_matrix[current_node, unvisited_nodes]
    d_dest = distance_matrix[destination_node, unvisited_nodes]
    scores = d_cur + d_dest

    # Slight tie-break: among near-equal insertion costs, prefer the closer
    # node to current (keeps the tour locally short and stable).
    # We add a tiny weighted nearest-neighbour term so that exact ties in the
    # insertion cost are resolved sensibly without changing the main signal.
    scores = scores + 1e-9 * d_cur

    best_idx = int(np.argmin(scores))
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
