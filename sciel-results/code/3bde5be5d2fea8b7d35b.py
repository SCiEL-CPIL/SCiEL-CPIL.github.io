"""
Multi-Start NN + 2-Opt guided TSP constructive heuristic.

Strategy:
  1. On detecting a new instance, run Multi-Start Nearest-Neighbor (all starts)
     followed by 2-Opt to convergence to build a high-quality reference tour.
  2. Pre-compute the successor for each node in that reference tour.
  3. During construction, follow the reference tour's successor when possible;
     if the current node's reference successor is already visited, fall back to
     a local scoring rule (distance + small lookahead).
  4. This 'follow-the-best-tour' approach leverages the full 2-Opt quality
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
        # Find nearest unvisited
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
                # Calculate delta
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
                if delta < -1e-10:
                    # Reverse segment
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour


def _or_opt(distance_matrix, tour):
    """Improve a tour using Or-Opt (relocate 1, 2, or 3 consecutive cities)."""
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
                # Segment to move: tour[i], tour[(i+1)%n], ..., tour[(i+seg_len-1)%n]
                # These are the seg_len cities starting at position i
                a = tour[(i - 1) % n]
                b = tour[i]
                c = tour[(i + seg_len - 1) % n]
                d = tour[(i + seg_len) % n]
                
                if b == d or a == d or b == a:
                    continue
                
                # Cost before: a-b + c-d
                cost_before = distance_matrix[a][b] + distance_matrix[c][d]
                
                # Try inserting the segment between each pair (x, y) in the tour
                # We need to find positions where we can insert without overlap
                for j in range(n):
                    if _is_deadline_passed():
                        return tour
                    # Skip positions that overlap with the segment
                    if j >= i and j < i + seg_len:
                        continue
                    if j == (i + seg_len) % n:
                        continue
                    if j == (i - 1) % n:
                        continue
                    
                    x = tour[(j - 1) % n]
                    y = tour[j]
                    
                    # Check that x and y are not part of the segment being moved
                    if x in tour[i:(i + seg_len) % n + 1 if (i + seg_len) % n > i else 0]:
                        # Need to handle wrap-around properly
                        pass
                    
                    # Cost after: a-d + x-b + c-y
                    cost_after = distance_matrix[a][d] + distance_matrix[x][b] + distance_matrix[c][y]
                    
                    if cost_after < cost_before - 1e-10:
                        # Perform the relocation
                        segment = []
                        for k in range(seg_len):
                            segment.append(tour[(i + k) % n])
                        
                        # Remove segment from tour (handle wrap)
                        new_tour = []
                        for k in range(n):
                            if i <= k < i + seg_len:
                                continue
                            new_tour.append(tour[k])
                        
                        # Insert segment after position j in new_tour
                        # j was the original index; need to find where to insert
                        # After removal, the position shifts
                        insert_pos = 0
                        for k in range(len(new_tour)):
                            if new_tour[k] == y:
                                insert_pos = k + 1
                                break
                        
                        new_tour[insert_pos:insert_pos] = segment
                        tour = new_tour
                        improved = True
                        break  # Restart scan for this seg_len
                if improved:
                    break
            if improved:
                break
    return tour


def _build_reference_tour(distance_matrix):
    """
    Build a high-quality reference tour using Multi-Start NN + 2-Opt.
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
        
        # Apply 2-opt then Or-Opt to this tour
        tour_2opt = _two_opt(distance_matrix, tour)
        tour_2opt = _or_opt(distance_matrix, tour_2opt)
        tour_len_2opt = sum(distance_matrix[tour_2opt[i]][tour_2opt[(i + 1) % n]] for i in range(n))
        
        if tour_len_2opt < best_len:
            best_len = tour_len_2opt
            best_tour = tour_2opt
    
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
    
    # Fallback: local scoring with lookahead toward destination
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    scores = dists_from_current.copy()
    
    # Add a penalty for nodes that are far from the destination
    if n_unvisited > 2:
        dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
        scores += 0.15 * dists_to_dest
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
