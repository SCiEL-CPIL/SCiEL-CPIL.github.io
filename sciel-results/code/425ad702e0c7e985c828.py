"""Phase-aware TSP constructive heuristic with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 115.0  # wall-clock safety margin under the 120 s limit
_T0 = time.monotonic()


def _remaining():
    return _DEADLINE - (time.monotonic() - _T0)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Hard safety: if we are near the wall-clock limit, fall back to O(n) nearest.
    if _remaining() < 0.5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_un = len(unvisited_nodes)
    if n_un == 0:
        return int(current_node)
    if n_un == 1:
        return int(unvisited_nodes[0])

    # Distances from current to all unvisited, and from each unvisited to destination.
    d_cur = distance_matrix[current_node, unvisited_nodes]
    d_dest = distance_matrix[destination_node, unvisited_nodes]

    # Phase: when few nodes remain, weight the closing edge heavily.
    if n_un <= 10:
        # Destination-aware: minimize d_cur + alpha * d_dest (alpha grows as we close).
        alpha = 1.0 + (10 - n_un) * 0.15
        scores = d_cur + alpha * d_dest
        return int(unvisited_nodes[np.argmin(scores)])

    # Middle phase: limited 2-step lookahead (cheapest 2nd hop) blended with nearest.
    # Keep it cheap: only consider a small candidate set.
    k = min(8, n_un)
    cand_idx = np.argpartition(d_cur, k - 1)[:k]
    cand_idx = cand_idx[np.argsort(d_cur[cand_idx])]
    cand = unvisited_nodes[cand_idx]

    # For each candidate, estimate the cheapest onward edge to any other unvisited node.
    # Restrict onward search to the same small candidate set plus destination for speed.
    # This approximates a 2-opt-ish local quality without full O(n^2) per step.
    onward_pool = np.concatenate([cand, [destination_node]])
    onward_pool = np.unique(onward_pool)

    best_score = np.inf
    best_node = int(cand[0])
    for ci in cand:
        # onward: min distance from ci to any node in onward_pool (excluding itself)
        od = distance_matrix[ci, onward_pool]
        od = np.delete(od, np.where(onward_pool == ci)[0])
        onward_min = od.min() if od.size else 0.0
        score = d_cur[np.where(cand == ci)[0][0]] + 0.5 * onward_min
        if score < best_score:
            best_score = score
            best_node = int(ci)

    return best_node
# EVOLVE-BLOCK-END
