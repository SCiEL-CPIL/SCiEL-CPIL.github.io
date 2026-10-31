"""
Multi-Start NN + 2-Opt + Or-Opt guided TSP constructive heuristic.

Strategy:
  1. On detecting a new instance, run Multi-Start Nearest-Neighbor (all starts)
     followed by 2-Opt then Or-Opt to convergence to build a high-quality reference tour.
  2. Pre-compute the successor for each node in that reference tour.
  3. During construction, follow the reference tour's successor when possible;
     if the current node's reference successor is already visited, fall back to
     a local scoring rule (distance + small lookahead).
  4. This 'follow-the-best-tour' approach leverages full local-search quality
     without needing to solve the TSP online.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 110.0  # Safety margin before 120s limit

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


def _two_opt(distance_matrix, tour):
    """Improve a tour using 2-opt until no improvement is found."""
    n = len(tour)
    if n < 4:
        return tour

    tour = list(tour)
    improved = True

    while improved and not _is_deadline_passed():
        improved = False
        for i in range(n - 2):
            if _is_deadline_passed():
                return tour
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
                if delta < -1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour


def _or_opt(distance_matrix, tour):
    """Improve a tour using Or-Opt: relocate segments of 1-3 cities."""
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
                # Segment to move: tour[i] ... tour[i+seg_len-1] (mod n)
                seg = [tour[(i + k) % n] for k in range(seg_len)]
                before = tour[(i - 1) % n]
                after = tour[(i + seg_len) % n]

                # Current cost of edges touching the segment
                cost_before = distance_matrix[before][seg[0]] + distance_matrix[seg[-1]][after]

                # Try inserting the segment at every valid position j
                # Position j means: insert between tour[(j-1)%n] and tour[j%n]
                # in the tour *with the segment removed*
                # But to keep it simple, we work with the original tour indices
                # and skip positions that overlap with the segment or are adjacent.

                for j in range(n):
                    if _is_deadline_passed():
                        return tour
                    # Skip if j is within the segment
                    in_seg = False
                    for k in range(seg_len):
                        if (i + k) % n == j:
                            in_seg = True
                            break
                    if in_seg:
                        continue
                    # Skip if j is adjacent to the segment (would recreate same or trivial move)
                    if j == i or j == (i - 1) % n or j == (i + seg_len) % n:
                        continue
                    # Also skip if the insertion position is "inside" the segment range
                    # (already handled by in_seg check for exact positions,
                    #  but also skip if j is right after the segment ends)

                    # Compute new cost:
                    # Remove edges: (before, seg[0]) and (seg[-1], after)
                    # Add edges: (before, after) and (insert_before, seg[0]) and (seg[-1], insert_after)
                    # But we need to be careful: when we remove the segment, the "after"
                    # node's predecessor changes. Let's compute directly.

                    # In the original tour, the edges are:
                    # ... before -> seg[0] -> ... -> seg[-1] -> after ...
                    # After removing seg: before -> after
                    # After inserting seg at position j:
                    #   We insert between tour[(j-1)%n] and tour[j%n] in the *remaining* tour.
                    #   But since we're working with the original tour, let's just compute
                    #   the delta by identifying which edges change.

                    # Simpler approach: compute the total length before and after.
                    # For speed, compute delta directly:
                    # Original edges that change:
                    #   (before, seg[0]), (seg[-1], after)
                    # New edges:
                    #   (before, after), (insert_before, seg[0]), (seg[-1], insert_after)
                    # But insert_before and insert_after depend on where j is relative to i.

                    # Let's just compute the full tour length for the candidate.
                    # This is O(n) per candidate, but with n=50 and seg_len small,
                    # the total is manageable within the deadline.

                    # Build candidate tour:
                    # Remove seg from position i, insert at position j (in the shortened tour)
                    # This is tricky with modular indices. Let's use a different approach:
                    # Create a new list.

                    # Actually, for correctness and simplicity, let's just compute the
                    # delta using the edge changes directly.

                    # Case analysis based on relative positions is complex.
                    # Instead, compute the candidate tour length directly.

                    # Build new tour by removing seg and inserting at j.
                    # Original tour: [t0, t1, ..., t_{n-1}]
                    # Remove positions i, i+1, ..., i+seg_len-1 (mod n)
                    # Insert seg between position j-1 and j (in the original indexing,
                    # but adjusted for removal)

                    # This is getting complex. Let's use a straightforward approach:
                    # Create a new list.
                    new_tour = []
                    for k in range(n):
                        if k != i and k != (i + 1) % n and (seg_len == 1 or (k != (i + 2) % n if seg_len >= 3 else True)):
                            # This is getting messy. Let me use a cleaner method.
                            pass

                    # Cleaner: build the tour without the segment, then insert.
                    # Step 1: Create tour without segment
                    without_seg = []
                    for k in range(n):
                        if k not in [(i + m) % n for m in range(seg_len)]:
                            without_seg.append(tour[k])

                    # Step 2: Find insertion point in without_seg
                    # We want to insert between the node that was at position j-1
                    # and the node at position j in the ORIGINAL tour, but only
                    # if those nodes are still in without_seg.
                    # Actually, the position j in the original tour corresponds to
                    # a specific pair of nodes. Let's find where to insert.

                    # The insertion is between tour[(j-1)%n] and tour[j%n] in the original.
                    # But if those nodes are in the segment, skip (already handled).
                    # In without_seg, find the index of tour[(j-1)%n] and tour[j%n].

                    # For simplicity, let's just try all insertion positions in without_seg.
                    # without_seg has n - seg_len elements.
                    # Try inserting at each position 0..n-seg_len.

                    for ins_pos in range(len(without_seg) + 1):
                        if _is_deadline_passed():
                            return tour
                        candidate = without_seg[:ins_pos] + seg + without_seg[ins_pos:]
                        if len(candidate) != n:
                            continue
                        # Compute length
                        cand_len = 0.0
                        for k in range(n):
                            cand_len += distance_matrix[candidate[k]][candidate[(k + 1) % n]]
                        orig_len = 0.0
                        for k in range(n):
                            orig_len += distance_matrix[tour[k]][tour[(k + 1) % n]]
                        if cand_len < orig_len - 1e-10:
                            tour = candidate
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break

    return tour


def _build_reference_tour(distance_matrix):
    """
    Build a high-quality reference tour using Multi-Start NN + 2-Opt + Or-Opt.
    Returns an array where reference_successor[i] is the node visited after i in the tour.
    """
    global _reference_successor

    n = distance_matrix.shape[0]
    if n <= 1:
        return np.array([], dtype=int)
    if n == 2:
        return np.array([1, 0])

    best_tour = None
    best_len = float('inf')

    # Multi-start: try all starting nodes
    for start in range(n):
        if _is_deadline_passed():
            break

        # Build NN tour from this start
        tour = _nn_tour(distance_matrix, start)

        # Compute tour length
        tour_len = sum(distance_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))

        if tour_len < best_len:
            best_len = tour_len
            best_tour = tour

        # Apply 2-opt to this tour
        tour_2opt = _two_opt(distance_matrix, tour)
        tour_len_2opt = sum(distance_matrix[tour_2opt[i]][tour_2opt[(i + 1) % n]] for i in range(n))

        if tour_len_2opt < best_len:
            best_len = tour_len_2opt
            best_tour = tour_2opt

        # Apply Or-Opt to the 2-opt result
        tour_oropt = _or_opt(distance_matrix, tour_2opt)
        tour_len_oropt = sum(distance_matrix[tour_oropt[i]][tour_oropt[(i + 1) % n]] for i in range(n))

        if tour_len_oropt < best_len:
            best_len = tour_len_oropt
            best_tour = tour_oropt

    if best_tour is None:
        # Fallback: simple NN from node 0
        best_tour = _nn_tour(distance_matrix, 0)

    # Build successor array
    successor = np.zeros(n, dtype=int)
    for i in range(n):
        successor[best_tour[i]] = best_tour[(i + 1) % n]

    _reference_successor = successor
    return successor


def _detect_new_instance(distance_matrix, unvisited_nodes):
    """Detect if we're starting a new instance."""
    global _last_dm_checksum

    n_total = distance_matrix.shape[0]
    n_unvisited = len(unvisited_nodes)

    # A new instance starts when all (or nearly all) nodes are unvisited
    if n_unvisited >= n_total - 1:
        # Quick checksum
        step = max(1, n_total // 10)
        checksum = np.sum(distance_matrix[::step, ::step])

        if _last_dm_checksum is None or abs(checksum - _last_dm_checksum) > 1e-6:
            _last_dm_checksum = checksum
            return True

    return False


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.

    Follows the pre-computed reference tour's successor when possible,
    falling back to local scoring when the reference successor is unavailable.
    """
    global _instance_initialized, _reference_successor

    n_unvisited = len(unvisited_nodes)

    if n_unvisited == 0:
        return -1

    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Detect new instance
    is_new_instance = _detect_new_instance(distance_matrix, unvisited_nodes)

    if is_new_instance:
        _instance_initialized = False
        _reference_successor = None
        # Build reference tour
        _build_reference_tour(distance_matrix)
        _instance_initialized = True

    if _is_deadline_passed():
        # Fallback: pure nearest neighbor
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])

    # Try to follow the reference tour
    if _reference_successor is not None:
        ref_next = int(_reference_successor[current_node])
        # Check if ref_next is in unvisited_nodes
        if ref_next in unvisited_nodes:
            return ref_next

    # Fallback: local scoring with small lookahead
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    scores = dists_from_current.copy()

    # Add a small penalty for nodes that are far from the destination
    if n_unvisited > 2:
        dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
        scores += 0.05 * dists_to_dest

    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
