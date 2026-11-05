"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Module-level deadline guard: start the clock on first use.
_T0 = None
_DEADLINE = 150.0 - 5.0  # 5s safety margin before the 150s wall-clock kill


def _remaining_time():
    global _T0
    if _T0 is None:
        _T0 = time.time()
    return _DEADLINE - (time.time() - _T0)


def _nearest_node_fallback(current_node, unvisited_nodes, distance_matrix):
    return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unvisited_nodes = np.asarray(unvisited_nodes)
    n = unvisited_nodes.size
    if n == 0:
        # Should not happen in a valid tour build; return current to be safe.
        return int(current_node)
    if n == 1:
        return int(unvisited_nodes[0])

    # If time is running out, fall back to cheap nearest-neighbour.
    if _remaining_time() < 0.05:
        return _nearest_node_fallback(current_node, unvisited_nodes, distance_matrix)

    # Immediate edge cost: current -> candidate
    d_cur = distance_matrix[current_node, unvisited_nodes]
    # Closure proxy: candidate -> destination (final return leg)
    d_dest = distance_matrix[unvisited_nodes, destination_node]

    # Smooth surrogate for min over next-hop: soft-min via log-sum-exp
    # soft_min(x) = -logsumexp(-x). We use a temperature to control smoothness.
    # For the "min next-hop" proxy, we want a smooth min of d_cur (cheap lookahead).
    # Use a moderate temperature so near-ties are smoothed but the minimum dominates.
    temp = 0.25  # temperature for soft-min (smaller = sharper)
    # soft_min of d_cur over candidates (vector of scalars)
    neg = -d_cur / temp
    m = np.max(neg)
    soft_min_cur = -(m + np.log(np.sum(np.exp(neg - m)))) * temp

    # Combine: immediate edge (dominant) + soft lookahead + closure to destination.
    # Weight immediate edge most heavily; closure second; soft-min as a smoother tie-break.
    score = d_cur + 0.5 * d_dest + 0.25 * soft_min_cur

    # Pick the candidate with the minimum combined score.
    idx = int(np.argmin(score))
    return int(unvisited_nodes[idx])
# EVOLVE-BLOCK-END
