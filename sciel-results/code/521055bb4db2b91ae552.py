"""Combined TSP constructive heuristic: reference-tour guidance with local optimization.

Strategy:
  1. On detecting a new instance, build a high-quality reference tour using
     Nearest-Neighbor followed by 2-Opt (and a light Or-Opt pass) to convergence.
  2. Pre-compute the successor for each node in that reference tour.
  3. During construction, follow the reference tour's successor when possible;
     if the current node's reference successor is already visited, fall back to
     a local scoring rule (distance + small lookahead + destination pull).
  4. For small remaining sets (<=5), use exact brute force to optimize the tail.
  5. Deadline guard: if time is running out, fall back to nearest neighbor.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import permutations

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

# Per-instance state
_last_dm_checksum = None
_reference_successor = None  # array: reference_successor[i] = next node after i in ref tour
_instance_initialized = False


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _nn_tour(distance_matrix, start):
    """Build a tour using nearest neighbor from a given start node."""
    n = distance_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    current = start

    for _ in range(n - 1):
        dists = distance_matrix[current].copy()
        dists[visited] = np.inf
        next_node = int(np.argmin(dists))
        tour.append(next_node)
        visited[next_node] = True
        current = next_node

    return tour


def _two_opt(distance_matrix, tour, max_passes=50):
    """Improve a tour using 2-opt until no improvement is found or max passes."""
    n = len(tour)
    if n < 4:
        return tour

    tour = list(tour)

    for _ in range(max_passes):
        if _is_deadline_passed():
            return tour
        improved = False
        for i in range(n - 2):
            if _is_deadline_passed():
                return tour
            a = tour[i]
            b = tour[i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c = tour[j]
                d = tour[(j + 1) % n]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                if delta < -1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return tour


def _or_opt(distance_matrix, tour):
    """Improve a tour using Or-Opt (relocate 1-3 city segments)."""
    n = len(tour)
    if n < 4:
        return tour

    tour = list(tour)
    improved = True

    while improved and not _is_deadline_passed():
        improved = False
        for seg_len in [1, 2, 3]:
            if _is_deadline_passed():
                return tour
            for i in range(n):
                if _is_deadline_passed():
                    return tour
                seg = tour[i:i + seg_len]
                if len(seg) < seg_len:
                    continue

                before = tour[(i - 1) % n]
                after = tour[(i + seg_len) % n]
                seg_start = seg[0]
                seg_end = seg[-1]

                # Try inserting at every other position
                for j in range(n):
                    if _is_deadline_passed():
                        return tour
                    # Skip positions that would recreate the same or adjacent
                    if j == i or j == (i - 1) % n or j == (i + seg_len) % n:
                        continue
                    # Skip if j is within the segment range
                    in_seg = False
                    for k in range(seg_len):
                        if (i + k) % n == j:
                            in_seg = True
                            break
                    if in_seg:
                        continue

                    insert_before = tour[(j - 1) % n]
                    insert_after = tour[j]

                    # Compute delta
                    # Remove edges: (before, seg_start), (seg_end, after)
                    # Add edges: (before, after), (insert_before, seg_start), (seg_end, insert_after)
                    # But we need to be careful about which edges are actually present
                    # For simplicity, compute the delta directly
                    delta = (distance_matrix[before][after] + distance_matrix[insert_before][seg_start] + distance_matrix[seg_end][insert_after]
                             - distance_matrix[before][seg_start] - distance_matrix[seg_end][after])
                    if delta < -1e-10:
                        # Perform the move
                        # Remove segment from position i
                        new_tour = tour[:i] + tour[i + seg_len:]
                        # Insert segment at position j (adjust for removal)
                        # If j > i, the index shifts by seg_len
                        if j > i:
                            insert_pos = j - seg_len
                        else:
                            insert_pos = j
                        new_tour = new_tour[:insert_pos] + seg + new_tour[insert_pos:]
                        tour = new_tour
                        improved = True
                        break
                if improved:
                    break
    return tour


def _build_reference_tour(distance_matrix, start_node):
    """Build a high-quality reference tour and compute successor mapping.
    
    Uses multi-start NN (try several starting points) followed by 2-Opt,
    keeping the best resulting tour.
    """
    global _reference_successor, _last_dm_checksum, _instance_initialized

    n = distance_matrix.shape[0]

    # Multi-start: try a few different starting nodes
    # Use start_node plus a few others spread around the circle
    candidates = [start_node]
    # Add a few other starting points for diversity
    for offset in [n // 4, n // 2, 3 * n // 4]:
        candidates.append((start_node + offset) % n)
    
    best_tour = None
    best_len = np.inf

    for s in candidates:
        if _is_deadline_passed():
            break
        tour = _nn_tour(distance_matrix, s)
        tour = _two_opt(distance_matrix, tour, max_passes=30)
        # Compute tour length
        tour_len = 0.0
        for i in range(n):
            tour_len += distance_matrix[tour[i], tour[(i + 1) % n]]
        if tour_len < best_len:
            best_len = tour_len
            best_tour = tour

    tour = best_tour

    # Apply Or-Opt for additional improvement
    if not _is_deadline_passed():
        tour = _or_opt(distance_matrix, tour)

    # One more 2-Opt pass after Or-Opt
    if not _is_deadline_passed():
        tour = _two_opt(distance_matrix, tour, max_passes=20)

    # Compute successor mapping
    _reference_successor = np.zeros(n, dtype=int)
    for i in range(n):
        _reference_successor[tour[i]] = tour[(i + 1) % n]

    # Store checksum to detect new instances
    _last_dm_checksum = hash(tuple(tuple(row) for row in distance_matrix))
    _instance_initialized = True

    return tour


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.

    Uses reference-tour guidance: builds a high-quality reference tour on detecting
    a new instance, then follows the reference tour's successor when possible.
    Falls back to local scoring if the reference successor is already visited.
    """
    global _last_dm_checksum, _reference_successor, _instance_initialized

    n_unvisited = len(unvisited_nodes)

    if n_unvisited == 0:
        return -1

    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # If few nodes left, use exact brute force to optimize the tail
    if n_unvisited <= 5:
        nodes = list(unvisited_nodes)
        best_cost = np.inf
        best_first_node = nodes[0]

        for perm in permutations(nodes):
            cost = distance_matrix[current_node, perm[0]]
            for i in range(len(perm) - 1):
                cost += distance_matrix[perm[i], perm[i + 1]]
            cost += distance_matrix[perm[-1], destination_node]

            if cost < best_cost:
                best_cost = cost
                best_first_node = perm[0]

        return int(best_first_node)

    # Detect new instance
    n_total = distance_matrix.shape[0]
    is_new_instance = (n_unvisited >= n_total - 1)  # First or second step of a new tour

    if is_new_instance:
        # Check if we've already built a reference for this instance
        current_checksum = hash(tuple(tuple(row) for row in distance_matrix))
        if _last_dm_checksum != current_checksum:
            _build_reference_tour(distance_matrix, current_node)

    # If deadline is close, use pure nearest neighbor
    if _is_deadline_passed():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])

    # Try to follow the reference tour's successor
    if _reference_successor is not None:
        ref_next = _reference_successor[current_node]
        if ref_next in unvisited_nodes:
            return int(ref_next)

    # Fallback: local scoring rule
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    scores = dists_from_current.copy()

    # Small destination pull
    alpha = 0.08
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    scores += alpha * dists_to_dest

    # 2-step lookahead
    if n_unvisited > 2:
        gamma = 0.12
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        np.fill_diagonal(sub_dist, np.inf)
        min_next = np.min(sub_dist, axis=1)
        scores += gamma * min_next

    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
