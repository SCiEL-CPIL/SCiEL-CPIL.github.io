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
                # Segment to move: tour[i], tour[i+1], ..., tour[i+seg_len-1]
                seg = tour[i:i + seg_len]
                if len(seg) < seg_len:
                    continue
                
                # Remove segment from position i
                # Tour becomes: ... tour[i-1] | tour[i+seg_len] ... tour[n-1] tour[0] ... tour[i-1]
                # We need to try inserting seg at every other position
                
                # Precompute neighbors for the segment
                seg_start = tour[i]
                seg_end = tour[(i + seg_len - 1) % n]
                before = tour[(i - 1) % n]
                after = tour[(i + seg_len) % n]
                
                # Cost of current edges around the segment
                cost_remove = distance_matrix[before][seg_start] + distance_matrix[seg_end][after]
                
                # Try inserting at every position j (not adjacent to original)
                for j in range(n):
                    if _is_deadline_passed():
                        return tour
                    # Skip positions that would recreate the same or adjacent
                    if j == i or j == (i - 1) % n or j == (i + seg_len) % n:
                        continue
                    if (i + seg_len - 1) % n == j or i == (j + seg_len) % n:
                        continue
                    # Also skip if j is within the segment range
                    for k in range(seg_len):
                        if (i + k) % n == j:
                            break
                    else:
                        # Position j means inserting between tour[(j-1)%n] and tour[j]
                        # But we need to account for the fact that we're removing seg first
                        # Let's just compute delta directly
                        insert_before = tour[(j - 1) % n]
                        insert_after = tour[j]
                        
                        # If inserting at position j, the new edges are:
                        # before -> seg_start (already counted in cost_remove for the "after" side)
                        # Actually let me think more carefully...
                        
                        # Current: before->seg_start ... seg_end->after
                        # After removal: before->after
                        # After insertion at j: insert_before->seg_start, seg_end->insert_after
                        # New cost: cost_remove - distance_matrix[before][after] + distance_matrix[insert_before][seg_start] + distance_matrix[seg_end][insert_after]
                        
                        # But wait, if j is in the remaining part, the "after" node changes
                        # This is getting complex. Let me use a simpler approach:
                        # Just compute the total tour length before and after.
                        
                        pass
                
                # Simpler approach: try each insertion position by computing delta
                for j in range(n):
                    if _is_deadline_passed():
                        return tour
                    if j == i:
                        continue
                    # Check if j is within the segment
                    in_seg = False
                    for k in range(seg_len):
                        if (i + k) % n == j:
                            in_seg = True
                            break
                    if in_seg:
                        continue
                    if j == (i + seg_len) % n or j == (i - 1) % n:
                        continue
                    
                    # Inserting seg between tour[(j-1)%n] and tour[j] in the tour WITHOUT the segment
                    # This is tricky because indices shift. Let me use a different approach.
                    
                    # Actually, let me just do a straightforward computation:
                    # Build the modified tour and compute the delta in edge costs
                    # Original edges affected: (before, seg_start), (seg_end, after)
                    # If j > i + seg_len: new edges: (before, after), (tour[(j-1)%n], seg_start), (seg_end, tour[j%n])
                    #   but tour[j%n] is now shifted... 
                    
                    # Let me just skip the complex case and do a simpler version:
                    # Only consider insertions where the segment is moved to a clearly different position
                    
                    # For simplicity, compute delta as:
                    # Remove edges: (before, seg_start) and (seg_end, after)
                    # Add edges: (before, after) and (insert_before, seg_start) and (seg_end, insert_after)
                    # But we need to be careful about which nodes are "before" and "after" the insertion point
                    # in the tour with the segment removed.
                    
                    # Let me just do it the brute force way for correctness:
                    new_tour = tour[:i] + tour[i + seg_len:]
                    # Insert seg at position j in new_tour
                    # j in original corresponds to j - seg_len if j > i + seg_len, or j if j < i
                    if j > i + seg_len:
                        insert_pos = j - seg_len
                    elif j < i:
                        insert_pos = j
                    else:
                        continue  # skip ambiguous cases
                    
                    if insert_pos < 0 or insert_pos > len(new_tour):
                        continue
                    
                    new_tour2 = new_tour[:insert_pos] + seg + new_tour[insert_pos:]
                    
                    # Compute delta in tour length
                    old_len = 0
                    for k in range(n):
                        old_len += distance_matrix[tour[k]][tour[(k + 1) % n]]
                    
                    new_len = 0
                    for k in range(n):
                        new_len += distance_matrix[new_tour2[k]][new_tour2[(k + 1) % n]]
                    
                    if new_len < old_len - 1e-10:
                        tour = new_tour2
                        improved = True
                        break  # restart scanning for this seg_len
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
        
        # Apply 2-opt to this tour
        tour_2opt = _two_opt(distance_matrix, tour)
        
        # Apply Or-Opt for additional improvement
        tour_oropt = _or_opt(distance_matrix, tour_2opt)
        # Final 2-opt pass after Or-Opt
        tour_final = _two_opt(distance_matrix, tour_oropt)
        
        tour_len_final = sum(distance_matrix[tour_final[i]][tour_final[(i + 1) % n]] for i in range(n))
        
        if tour_len_final < best_len:
            best_len = tour_len_final
            best_tour = tour_final
    
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
    
    # Add a penalty for nodes that are far from the destination
    if n_unvisited > 2:
        dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
        scores += 0.15 * dists_to_dest
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
