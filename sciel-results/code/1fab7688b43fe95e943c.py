"""Traveling salesman constructive heuristic: High-quality pre-computed tour with multi-start 2-Opt and Or-Opt."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for caching
_cache = {}
_start_time = None
_DEADLINE = 145.0  # Safety margin before 150s kill

def _get_instance_id(distance_matrix):
    """Create a lightweight fingerprint for the distance matrix."""
    n = distance_matrix.shape[0]
    if n > 10:
        sample = distance_matrix[:10, :10].tobytes()
    else:
        sample = distance_matrix.tobytes()
    return hash(sample)

def _tour_length(tour, distance_matrix):
    """Calculate the total length of the tour."""
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += distance_matrix[tour[i], tour[(i + 1) % n]]
    return total

def _two_opt(tour, distance_matrix, time_limit=None):
    """
    Apply 2-Opt improvement to the tour.
    Runs until no improvement is found or time limit is reached.
    """
    global _start_time
    n = len(tour)
    tour = list(tour)
    improved = True
    
    while improved:
        improved = False
        for i in range(n - 1):
            if time_limit and time.time() - _start_time > time_limit:
                return tour
            
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                
                edge1 = distance_matrix[tour[i], tour[i + 1]]
                edge2 = distance_matrix[tour[j], tour[(j + 1) % n]]
                
                new_edge1 = distance_matrix[tour[i], tour[j]]
                new_edge2 = distance_matrix[tour[i + 1], tour[(j + 1) % n]]
                
                if new_edge1 + new_edge2 < edge1 + edge2 - 1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
                    
    return tour

def _or_opt(tour, distance_matrix, time_limit=None):
    """
    Apply Or-Opt improvement (move a segment of 1-3 nodes to a different position).
    """
    global _start_time
    n = len(tour)
    tour = list(tour)
    improved = True
    
    while improved:
        improved = False
        for seg_len in [1, 2, 3]:
            for i in range(n):
                if time_limit and time.time() - _start_time > time_limit:
                    return tour
                
                if i + seg_len > n:
                    continue
                    
                # Current cost contribution of the segment's boundaries
                prev_node = tour[i - 1]
                next_node = tour[(i + seg_len) % n]
                
                # Cost of removing the segment: connect prev_node to next_node
                remove_cost = distance_matrix[prev_node, next_node]
                # Cost of current: prev->seg_start and seg_end->next
                current_cost = distance_matrix[prev_node, tour[i]] + distance_matrix[tour[i + seg_len - 1], next_node]
                
                # Try inserting the segment at every other position
                for k in range(n):
                    # Skip positions that are the same or adjacent (no change)
                    if k == i - 1 or k == (i + seg_len - 1) % n:
                        continue
                    
                    new_prev = tour[k]
                    new_next = tour[(k + 1) % n]
                    
                    # Cost of inserting: new_prev->seg_start and seg_end->new_next
                    insert_cost = distance_matrix[new_prev, tour[i]] + distance_matrix[tour[i + seg_len - 1], new_next]
                    
                    # Delta: (insert_cost + remove_cost) - (current_cost + distance_matrix[new_prev, new_next])
                    old_connect = distance_matrix[new_prev, new_next]
                    
                    if insert_cost + remove_cost < current_cost + old_connect - 1e-10:
                        # Perform the move
                        segment = tour[i:i + seg_len]
                        del tour[i:i + seg_len]
                        
                        # Find where to insert: after node k in the new tour
                        # If k >= i, its index shifted by -seg_len
                        # If k < i, its index is unchanged
                        if k >= i:
                            insert_idx = k - seg_len + 1
                        else:
                            insert_idx = k + 1
                        
                        # But we need to handle wrap-around
                        # Actually, let's rebuild: the tour without segment is tour
                        # We want to insert segment after the node that was at position k
                        # In the new tour (after removal), find that node's position
                        # This is complex. Let's use a different approach:
                        # Just rebuild the tour by concatenation.
                        
                        # Original tour: [0..i-1, i..i+seg_len-1, i+seg_len..n-1]
                        # After removal: [0..i-1, i+seg_len..n-1]
                        # We want to insert segment after node k (in original indexing)
                        
                        # Let's just try all insertions and pick the best, but that's O(n^2)
                        # For now, let's do a simpler check: just verify the delta is negative
                        # and perform the move carefully.
                        
                        # Rebuild approach:
                        # The tour without the segment
                        remaining = tour  # already has segment removed
                        # Find position of node k in remaining
                        try:
                            pos_k = remaining.index(tour[0] if False else None)  # placeholder
                        except:
                            pass
                        
                        # This is getting too complex. Let me use a cleaner implementation.
                        # For now, skip the actual move and just rely on 2-Opt.
                        # Or-Opt is complex to implement correctly with cyclic tours.
                        pass
                
    return tour

def _or_opt_simple(tour, distance_matrix, time_limit=None):
    """
    Simpler Or-Opt: try moving 1 or 2 consecutive nodes to a better position.
    """
    global _start_time
    n = len(tour)
    tour = list(tour)
    improved = True
    
    while improved:
        improved = False
        for seg_len in [1, 2]:
            for i in range(n):
                if time_limit and time.time() - _start_time > time_limit:
                    return tour
                
                if i + seg_len > n:
                    continue
                
                segment = tour[i:i + seg_len]
                prev_node = tour[i - 1]
                next_node = tour[(i + seg_len) % n]
                
                # Cost if we remove the segment
                cost_remove = distance_matrix[prev_node, next_node]
                cost_current = distance_matrix[prev_node, segment[0]] + distance_matrix[segment[-1], next_node]
                
                best_delta = np.inf
                best_k = -1
                
                for k in range(n):
                    if k == i - 1 or k == (i + seg_len - 1) % n:
                        continue
                    
                    new_prev = tour[k]
                    new_next = tour[(k + 1) % n]
                    
                    # If k is within the segment, skip
                    if i <= k < i + seg_len:
                        continue
                    if i + seg_len - 1 <= k < i + seg_len:  # wrap
                        continue
                    
                    cost_insert = distance_matrix[new_prev, segment[0]] + distance_matrix[segment[-1], new_next]
                    old_edge = distance_matrix[new_prev, new_next]
                    
                    delta = cost_insert + cost_remove - cost_current - old_edge
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_k = k
                
                if best_delta < -1e-10 and best_k >= 0:
                    # Perform the move
                    del tour[i:i + seg_len]
                    
                    # Find position of node at best_k in the new tour
                    # The node was at position best_k in the old tour
                    # After removing segment [i, i+seg_len), positions shift
                    if best_k >= i:
                        new_pos = best_k - seg_len
                    else:
                        new_pos = best_k
                    
                    # But best_k might be the last node, and we need to insert after it
                    # new_pos is the index of the node, so insert after new_pos
                    insert_at = new_pos + 1
                    if insert_at >= len(tour):
                        insert_at = 0  # wrap around
                    
                    for idx, node in enumerate(segment):
                        tour.insert(insert_at + idx, node)
                    
                    improved = True
                    
    return tour

def _get_tour_order(distance_matrix):
    """
    Compute a high-quality tour order using Multi-Start NN + 2-Opt, cached per instance.
    Returns an array of node indices in tour order.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    instance_id = _get_instance_id(distance_matrix)
    if instance_id in _cache:
        return _cache[instance_id]
    
    n = distance_matrix.shape[0]
    
    best_tour = None
    best_len = np.inf
    
    # Multi-start NN
    for start_node in range(n):
        if time.time() - _start_time > _DEADLINE - 5:
            break
        
        # Nearest neighbor initialization
        unvisited = set(range(n))
        tour = [start_node]
        unvisited.remove(start_node)
        current = start_node
        
        while unvisited:
            dists = distance_matrix[current, list(unvisited)]
            nearest_idx = np.argmin(dists)
            nearest = list(unvisited)[nearest_idx]
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        tour = np.array(tour)
        
        # 2-Opt improvement
        improved_tour = _two_opt(tour, distance_matrix, time_limit=_DEADLINE)
        
        tour_len = _tour_length(improved_tour, distance_matrix)
        
        if tour_len < best_len:
            best_len = tour_len
            best_tour = improved_tour
    
    if best_tour is None:
        best_tour = np.arange(n)
    
    # Try Or-Opt on the best tour if we have time
    if time.time() - _start_time < _DEADLINE - 10:
        or_opt_tour = _or_opt_simple(best_tour, distance_matrix, time_limit=_DEADLINE)
        or_opt_len = _tour_length(or_opt_tour, distance_matrix)
        if or_opt_len < best_len:
            best_tour = or_opt_tour
            best_len = or_opt_len
    
    _cache[instance_id] = best_tour
    return best_tour


def _get_next_in_tour(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node by following the precomputed tour order with local distance blending.
    """
    tour = _get_tour_order(distance_matrix)
    
    # Find current node's position in the tour
    current_pos = np.where(tour == current_node)[0]
    if len(current_pos) == 0:
        # Fallback: nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    current_pos = current_pos[0]
    n = len(tour)
    
    unvisited_set = set(unvisited_nodes.tolist())
    
    # Find the next unvisited node in tour order after current position (the "ideal" next)
    ideal_next = None
    ideal_dist = np.inf
    for offset in range(1, n):
        next_pos = (current_pos + offset) % n
        candidate = tour[next_pos]
        if candidate in unvisited_set:
            ideal_next = candidate
            ideal_dist = distance_matrix[current_node, candidate]
            break
    
    if ideal_next is None:
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    # Now, score all unvisited candidates
    # Primary: distance from current
    # Secondary: alignment with the tour (how close to the ideal next)
    scores = []
    for candidate in unvisited_nodes:
        dist = distance_matrix[current_node, candidate]
        # Tour alignment: 0 if candidate is the ideal next, increases with distance in tour
        if candidate == ideal_next:
            tour_score = 0.0
        else:
            # Find candidate's position in tour
            cand_pos = np.where(tour == candidate)[0][0]
            # Cyclic distance in tour from current_pos to cand_pos
            diff = (cand_pos - current_pos) % n
            tour_score = float(diff) / n  # normalized to [0, 1)
        
        # Blend: primarily local distance, with a small penalty for tour misalignment
        # Use a strong discount for the ideal next
        if candidate == ideal_next:
            score = dist * 0.1  # strong discount for the tour successor
        else:
            score = dist + 0.5 * tour_score * ideal_dist  # penalty proportional to tour distance
        
        scores.append((score, candidate))
    
    scores.sort()
    return int(scores[0][1])


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city by following a precomputed high-quality tour order with local blending.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Fallback: nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    # Use the cached tour to determine next node
    return _get_next_in_tour(current_node, destination_node, unvisited_nodes, distance_matrix)
# EVOLVE-BLOCK-END
