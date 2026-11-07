"""Heuristic matrix for ACO-TSP: High-quality tour-based heuristic."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: 330s wall-clock limit, use 300s as safety margin.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0


def _two_opt(tour, dist_matrix):
    """Improve a tour using 2-opt heuristic. Returns the improved tour."""
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            if time.time() > _DEADLINE - 10.0:
                return tour
            for j in range(i + 1, n):
                if j == i + 1:
                    continue
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                
                current_cost = dist_matrix[a, b] + dist_matrix[c, d]
                new_cost = dist_matrix[a, c] + dist_matrix[b, d]
                
                if new_cost < current_cost - 1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    return tour

def _or_opt(tour, dist_matrix):
    """Improve a tour using Or-opt heuristic (move segments of length 1, 2, 3)."""
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for seg_len in [1, 2, 3]:
            if time.time() > _DEADLINE - 10.0:
                return tour
            for i in range(n):
                # Segment to move: tour[i] ... tour[i+seg_len-1]
                # Neighbors: tour[i-1] and tour[i+seg_len]
                prev_node = tour[(i - 1) % n]
                next_node = tour[(i + seg_len) % n]
                
                # Cost of removing segment:
                # Edges (prev_node, tour[i]) and (tour[i+seg_len-1], next_node)
                # New edges: (prev_node, next_node)
                cost_remove = dist_matrix[prev_node, tour[i]] + dist_matrix[tour[i + seg_len - 1], next_node]
                cost_connect = dist_matrix[prev_node, next_node]
                
                # Try inserting segment at every other position k
                for k in range(n):
                    # Skip if k is close to i (to avoid trivial moves or invalid overlaps)
                    # The segment occupies indices i to i+seg_len-1.
                    # We are removing it. The tour shrinks by seg_len.
                    # We want to insert it between tour[k-1] and tour[k] in the *remaining* tour?
                    # It's easier to think in terms of the original indices.
                    
                    # If we insert between k-1 and k, we need k to not be in the segment [i, i+seg_len-1]
                    # and k-1 to not be in the segment.
                    
                    # Indices in the segment: i, i+1, ..., i+seg_len-1
                    # k is the index after which we insert. So the new edge is (tour[k-1], tour[i]) and (tour[i+seg_len-1], tour[k]).
                    # Wait, standard Or-opt:
                    # Remove segment S from between A and B.
                    # Insert S between C and D.
                    # New edges: (A, B), (C, S_start), (S_end, D).
                    # Old edges: (A, S_start), (S_end, B), (C, D).
                    
                    # In our loop:
                    # A = prev_node, B = next_node.
                    # S_start = tour[i], S_end = tour[i+seg_len-1].
                    # C = tour[k-1], D = tour[k].
                    
                    # We must ensure the insertion point is not "inside" the removed segment.
                    # The removed segment is at indices i ... i+seg_len-1.
                    # The gap is between i-1 and i+seg_len.
                    # We can insert at any k such that k is not in [i, i+seg_len-1] and k-1 is not in [i, i+seg_len-1].
                    # Also, we shouldn't put it back where it was.
                    
                    k_mod = k % n
                    # Check if k or k-1 is in the segment
                    in_seg_k = False
                    in_seg_km1 = False
                    for m in range(seg_len):
                        idx = (i + m) % n
                        if idx == k_mod:
                            in_seg_k = True
                        if idx == (k - 1) % n:
                            in_seg_km1 = True
                    if in_seg_k or in_seg_km1:
                        continue
                    
                    # Also skip if it's the same position (A, B)
                    if (k - 1) % n == (i - 1) % n and k_mod == (i + seg_len) % n:
                        continue

                    C = tour[(k - 1) % n]
                    D = tour[k]
                    
                    cost_insert_old = dist_matrix[C, D]
                    cost_insert_new = dist_matrix[C, tour[i]] + dist_matrix[tour[i + seg_len - 1], D]
                    
                    delta = (cost_connect + cost_insert_new) - (cost_remove + cost_insert_old)
                    
                    if delta < -1e-10:
                        # Perform move
                        # 1. Remove segment
                        segment = tour[i:i+seg_len]
                        # Be careful with list slicing if it wraps around? 
                        # Standard Or-opt assumes non-wrapping or handles it. 
                        # Since we use modulo for neighbors, but list slicing doesn't wrap easily.
                        # Let's assume the segment does not wrap around the end of the list for simplicity in slicing.
                        # If i+seg_len > n, we need to handle wrapping.
                        # For N=50, wrapping is rare but possible.
                        
                        if i + seg_len <= n:
                            new_tour = tour[:i] + tour[i+seg_len:]
                            # Find insertion point in new_tour
                            # k was index in old tour. 
                            # If k > i, the index in new_tour is k - seg_len.
                            # If k < i, the index is k.
                            # But we removed i..i+seg_len-1.
                            # Let's just reconstruct by identifying nodes.
                            
                            # A simpler robust way:
                            # Create a set of nodes in the segment
                            seg_nodes = set(tour[i:i+seg_len]) if i + seg_len <= n else set(tour[i:] + tour[:i%seg_len])
                            
                            # Actually, let's just use a helper to remove and insert by node value or index carefully.
                            # Given N=50, we can afford a slightly less optimized move if it's correct.
                            
                            # Let's stick to the non-wrapping case for the slice, which is 99% of cases.
                            # If i + seg_len > n, we break the tour into two parts.
                            if i + seg_len <= n:
                                remaining = tour[:i] + tour[i+seg_len:]
                                # Determine insertion index in 'remaining'
                                # If k > i, the node at k in old tour is now at k-seg_len in remaining?
                                # No, indices shift.
                                # Let's find the index of D in remaining.
                                # D is tour[k].
                                # If k > i, index is k - seg_len.
                                # If k < i, index is k.
                                if k > i:
                                    ins_idx = k - seg_len
                                else:
                                    ins_idx = k
                                # Insert segment before D (which is at ins_idx)
                                # So insert at ins_idx
                                new_tour = remaining[:ins_idx] + segment + remaining[ins_idx:]
                            else:
                                # Wrapping case
                                # This is complex. For N=50, we can just use a generic remove/insert by value.
                                # But tour is a list of unique indices.
                                pass # Fallback to no move for simplicity if wrapping, or handle properly.
                                # Let's handle it properly by creating a new list.
                                # Remove nodes in segment from tour
                                seg_set = set(tour[i:(i+seg_len)%n] if (i+seg_len)%n != 0 else tour[i:]) 
                                # This modulo logic is tricky.
                                # Let's just say if it wraps, we skip Or-opt for that segment to avoid bugs.
                                # 2-opt will still work.
                                continue

                            tour = new_tour
                            improved = True
                            break # Restart inner loop for this seg_len
                if improved:
                    break
            if improved:
                break
    return tour

def heuristics(distance_matrix):
    """Return an edge-attractiveness matrix (larger = more attractive).

    Strategy:
    1. Generate a Nearest Neighbor (NN) tour.
    2. Improve it with 2-opt and Or-opt local search.
    3. Create a heuristic matrix where edges in this "good" tour have extremely
       high attractiveness, and other edges have lower attractiveness based on
       inverse distance.
    """
    if time.time() > _DEADLINE - 5.0:
        d = np.asarray(distance_matrix, dtype=np.float64)
        return 1.0 / (d + 1e-9) ** 2

    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # 1. Nearest Neighbor Tour
    visited = [False] * n
    tour = [0]
    visited[0] = True
    current = 0
    
    for _ in range(n - 1):
        if time.time() > _DEADLINE - 10.0:
            break
        min_dist = float('inf')
        next_node = -1
        for j in range(n):
            if not visited[j]:
                if d[current, j] < min_dist:
                    min_dist = d[current, j]
                    next_node = j
        if next_node == -1:
            break
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    if len(tour) < n:
        heuristic = 1.0 / (d + 1e-9) ** 2
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # 2. Local Search Improvement
    # Alternate between 2-opt and Or-opt
    for _ in range(10): # Max 10 rounds
        if time.time() > _DEADLINE - 10.0:
            break
        tour = _two_opt(tour, d)
        tour = _or_opt(tour, d)

    # 3. Construct Heuristic Matrix
    # Base heuristic: inverse distance squared
    base_heuristic = 1.0 / (d + 1e-9) ** 2
    
    # Topological boost: edges in the improved tour get a HUGE multiplier
    # This ensures the ACO strongly follows the structure of the good tour
    final_heuristic = base_heuristic.copy()
    
    # Identify edges in the tour
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        # Set to a very large value
        final_heuristic[u, v] = 1e6
        final_heuristic[v, u] = 1e6
    
    # Ensure diagonal is 0 (evaluator clamps non-positive to 1e-9)
    np.fill_diagonal(final_heuristic, 0.0)
    
    # Ensure all values are finite and positive
    final_heuristic = np.where(np.isfinite(final_heuristic) & (final_heuristic > 0),
                               final_heuristic, 1e-9)
    
    return final_heuristic
# EVOLVE-BLOCK-END
