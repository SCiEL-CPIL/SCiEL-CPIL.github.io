"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 138.0
_DEADLINE_MARGIN = 0.75

def _time_left():
    return max(0.0, TIME_LIMIT - (time.time() - _START_TIME))

def _fast_next(current_node, destination_node, unvisited_nodes, distance_matrix):
    U = np.asarray(unvisited_nodes, dtype=int)
    if len(U) == 0:
        return -1
    if len(U) == 1:
        return int(U[0])

    d = distance_matrix[current_node, U]
    # If destination is still unvisited, avoid ending too early by adding a small destination penalty.
    if destination_node in U:
        # Prefer nodes that keep the path from terminating at destination before all nodes are visited.
        # A tiny penalty is enough; the main signal remains nearest neighbor.
        d = d + 1e-9 * (U != destination_node)
    return int(U[np.argmin(d)])

def _two_opt_path(nodes, distance_matrix, deadline):
    """
    Improve an open path: nodes[0] -> ... -> nodes[-1].
    Only performs 2-opt moves that reduce the path length and are safe within the deadline.
    """
    n = len(nodes)
    if n < 4:
        return nodes

    nodes = list(nodes)
    def segment_cost(a, b, c, d):
        return distance_matrix[a, b] + distance_matrix[c, d]

    improved = True
    while improved and _time_left() > deadline:
        improved = False
        # Forward pass
        for i in range(n - 2):
            if _time_left() <= deadline:
                break
            ai = nodes[i]
            bi = nodes[i + 1]
            best_j = -1
            best_gain = 0.0
            # Try reversing nodes[i+1..j]
            for j in range(i + 2, n):
                if _time_left() <= deadline:
                    break
                cj = nodes[j]
                dj = nodes[j + 1] if j + 1 < n else None
                if dj is None:
                    old = distance_matrix[ai, bi]
                    new = distance_matrix[ai, cj]
                    gain = old - new
                else:
                    old = distance_matrix[ai, bi] + distance_matrix[cj, dj]
                    new = distance_matrix[ai, cj] + distance_matrix[bi, dj]
                    gain = old - new
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_j = j
            if best_j != -1:
                # Reverse nodes[i+1..best_j]
                nodes[i + 1:best_j + 1] = reversed(nodes[i + 1:best_j + 1])
                improved = True
        # Backward pass can help escape some local patterns, but keep it limited by time.
        if improved and _time_left() > deadline:
            for i in range(n - 3, -1, -1):
                if _time_left() <= deadline:
                    break
                ai = nodes[i]
                bi = nodes[i + 1]
                best_j = -1
                best_gain = 0.0
                for j in range(i + 2, n):
                    if _time_left() <= deadline:
                        break
                    cj = nodes[j]
                    dj = nodes[j + 1] if j + 1 < n else None
                    if dj is None:
                        old = distance_matrix[ai, bi]
                        new = distance_matrix[ai, cj]
                        gain = old - new
                    else:
                        old = distance_matrix[ai, bi] + distance_matrix[cj, dj]
                        new = distance_matrix[ai, cj] + distance_matrix[bi, dj]
                        gain = old - new
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_j = j
                if best_j != -1:
                    nodes[i + 1:best_j + 1] = reversed(nodes[i + 1:best_j + 1])
                    improved = True
    return nodes

def _construct_and_optimize(current_node, destination_node, unvisited_nodes, distance_matrix):
    U = np.asarray(unvisited_nodes, dtype=int)
    if len(U) <= 1:
        return _fast_next(current_node, destination_node, U, distance_matrix)

    # Initial construction: nearest neighbor from current, with destination kept unvisited if possible.
    remaining = set(int(x) for x in U)
    path = [int(current_node)]
    cur = int(current_node)

    while remaining:
        # If time is tight, stop constructing and use nearest neighbor for the final pick.
        if _time_left() <= deadline:
            break
        if len(remaining) == 1:
            nxt = next(iter(remaining))
        else:
            # Prefer nearest unvisited node, but if destination is the only remaining or would be chosen,
            # defer it until the end to keep an open path.
            dists = distance_matrix[cur, list(remaining)]
            # Add a tiny penalty to destination unless it is the last node.
            if destination_node in remaining and len(remaining) > 1:
                dists = dists.copy()
                idx_dest = np.where(np.asarray(list(remaining)) == destination_node)[0]
                if len(idx_dest):
                    dists[idx_dest] += 1e-6
            nxt = list(remaining)[int(np.argmin(dists))]
        path.append(int(nxt))
        remaining.remove(int(nxt))
        cur = int(nxt)

    # If we stopped early due to deadline, return the next node from the fast heuristic.
    if _time_left() <= deadline:
        return _fast_next(current_node, destination_node, U, distance_matrix)

    # Optimize the open path (excluding the starting current_node? The evaluator will append the returned node to the current tour.
    # We should optimize the suffix that will be used: current -> path[1:]
    # The first element of path is current_node and is fixed.
    suffix = path[1:]
    if len(suffix) >= 3:
        suffix = _two_opt_path(suffix, distance_matrix, deadline)

    # The next node to return is the first node after current in the optimized suffix.
    if not suffix:
        return _fast_next(current_node, destination_node, U, distance_matrix)
    return int(suffix[0])

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a global 2-Opt refinement of the remaining open path when time permits.
    Falls back to a fast nearest-neighbor choice near the deadline.
    """
    U = np.asarray(unvisited_nodes, dtype=int)
    if len(U) == 0:
        return -1
    if len(U) == 1:
        return int(U[0])

    deadline = _DEADLINE_MARGIN
    if _time_left() <= deadline:
        return _fast_next(current_node, destination_node, U, distance_matrix)

    # Use a small amount of the spare budget for global improvement.
    # The constructive phase is O(n^2), and 2-opt is O(n^2) per pass; for n<=50 this is cheap.
    # We cap the deadline to avoid overshooting the evaluator limit.
    deadline = min(deadline, max(0.05, _time_left() * 0.15))
    try:
        return _construct_and_optimize(current_node, destination_node, U, distance_matrix)
    except Exception:
        return _fast_next(current_node, destination_node, U, distance_matrix)
# EVOLVE-BLOCK-END
