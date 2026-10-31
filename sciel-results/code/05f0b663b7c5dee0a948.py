"""TSP constructive heuristic: NN skeleton + iterative 2-opt local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 112.0  # Safety margin before 120s limit

def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _two_opt(tour, dm):
    """In-place 2-opt improvement of the tour. Returns tour."""
    n = len(tour)
    if n < 4:
        return tour
    improved = True
    max_sweeps = 30
    sweep = 0
    while improved and sweep < max_sweeps and not _is_deadline_passed():
        improved = False
        sweep += 1
        for i in range(n - 2):
            if _is_deadline_passed():
                return tour
            a = tour[i]
            b = tour[i + 1]
            j2 = i + 2
            for j in range(j2, n):
                c = tour[j]
                d = tour[(j + 1) % n]
                gain = dm[a, c] + dm[b, d] - dm[a, b] - dm[c, d]
                if gain < -1e-12:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour


def _nearest_neighbor_tour(start, dm, n):
    """Build a nearest-neighbor tour starting at `start` and returning to it."""
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    current = start
    remaining = n - 1
    while remaining > 0:
        if _is_deadline_passed():
            break
        dists = dm[current]
        dists = dists.copy()
        dists[visited] = np.inf
        nxt = int(np.argmin(dists))
        visited[nxt] = True
        tour.append(nxt)
        current = nxt
        remaining -= 1
    return tour


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Structural change: build a complete NN tour from `current_node` back to
    `destination_node`, apply iterative 2-opt, then return the city that
    immediately follows `current_node` in the optimized tour.
    """
    n_unvisited = len(unvisited_nodes)
    current_node = int(current_node)
    destination_node = int(destination_node)

    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        c1 = int(unvisited_nodes[0])
        c2 = int(unvisited_nodes[1])
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c1, destination_node]
        return c1 if cost1 < cost2 else c2

    # Build full node list: current, unvisited..., destination
    full = np.concatenate(([current_node], unvisited_nodes, [destination_node]))
    n = len(full)
    if n < 4:
        # Degenerate: just nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    sub = distance_matrix[np.ix_(full, full)]

    # Nearest-neighbor tour starting at index 0 (current_node)
    tour = _nearest_neighbor_tour(0, sub, n)

    # Iterative 2-opt
    tour = _two_opt(tour, sub)

    # The next city after index 0 in the optimized tour
    next_idx = tour[1]
    return int(full[next_idx])

# EVOLVE-BLOCK-END
