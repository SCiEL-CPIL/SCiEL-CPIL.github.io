"""Constructive TSP heuristic using multi-start local search (2-Opt + Or-Opt)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

# Global cache for the optimized tour
_cached_dm_fingerprint = None
_cached_tour = None
_START_TIME = time.time()
_DEADLINE = 150.0
_TIME_MARGIN = 5.0  # Safety margin in seconds

def _get_fingerprint(dm):
    """Create a cheap fingerprint of the distance matrix to detect instance changes."""
    # Using a small subset to avoid full array hashing cost
    return float(np.sum(dm[:5, :5]) + np.sum(dm[-5:, -5:]))

def _tour_length(tour, dist_matrix):
    """Calculate the total length of a tour."""
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += dist_matrix[tour[i], tour[(i + 1) % n]]
    return total

def _nearest_neighbor_tour(n, dist_matrix, start_node=0):
    """Generate a tour using the Nearest Neighbor heuristic starting from start_node."""
    visited = [False] * n
    tour = [start_node]
    visited[start_node] = True
    current = start_node
    
    for _ in range(n - 1):
        min_dist = float('inf')
        next_node = -1
        for i in range(n):
            if not visited[i]:
                d = dist_matrix[current, i]
                if d < min_dist:
                    min_dist = d
                    next_node = i
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    return tour

def _random_tour(n):
    """Generate a random tour."""
    tour = list(range(n))
    random.shuffle(tour)
    return tour

def _two_opt(tour, dist_matrix):
    """Improve the tour using 2-Opt moves."""
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Skip if they are adjacent (i, i+1) or wrap-around (0, n-1)
                if i == 0 and j == n - 1:
                    continue
                
                # Calculate delta cost
                c1 = dist_matrix[tour[i], tour[i + 1]]
                c2 = dist_matrix[tour[j], tour[(j + 1) % n]]
                c3 = dist_matrix[tour[i], tour[j]]
                c4 = dist_matrix[tour[i + 1], tour[(j + 1) % n]]
                
                if c3 + c4 < c1 + c2 - 1e-9:
                    # Reverse the segment between i+1 and j
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour

def _or_opt(tour, dist_matrix):
    """Improve the tour using Or-Opt moves (moving 1, 2, or 3 nodes)."""
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for length in [1, 2, 3]:
            for i in range(n):
                # Segment is tour[i : i+length]
                # We try to insert it at position j
                for j in range(n):
                    # Skip trivial moves
                    if j == i or j == (i - 1) % n or j == (i + length) % n:
                        continue
                    
                    # Check if the move is valid (no overlap)
                    # The segment is at [i, i+length-1]
                    # We are inserting before index j in the new tour.
                    # We need to ensure the insertion point is not within the segment being moved.
                    # In the original tour, the segment occupies indices i to i+length-1.
                    # The insertion point j is an index in the original tour.
                    # If j is in [i, i+length-1], it's invalid.
                    if i <= j < i + length:
                        continue
                    
                    # Calculate delta cost
                    # Original edges:
                    # 1. (tour[i-1], tour[i])
                    # 2. (tour[i+length-1], tour[i+length])
                    # 3. (tour[j-1], tour[j])
                    # 4. (tour[j], tour[j+1]) -- Wait, Or-Opt removes a segment and inserts it.
                    
                    # Let's define the move precisely:
                    # Remove segment S = tour[i : i+length]
                    # Insert S before tour[j] in the remaining tour.
                    
                    # Indices in original tour:
                    prev_i = (i - 1) % n
                    next_i = (i + length) % n
                    prev_j = (j - 1) % n
                    next_j = j # We insert before j, so the edge (prev_j, j) is broken.
                    
                    # Wait, standard Or-Opt definition:
                    # Move segment from position i to position j.
                    # If we remove S, the neighbors are prev_i and next_i.
                    # The edge (prev_i, next_i) is formed.
                    # We insert S between prev_j and next_j? No, we insert S at position j.
                    # So the edge (prev_j, j) is broken, and (prev_j, S[0]) and (S[-1], j) are formed.
                    
                    # Let's verify indices.
                    # Original: ... prev_i, S, next_i ... prev_j, j, next_j ...
                    # New: ... prev_i, next_i ... prev_j, S, j, next_j ...
                    
                    # Edges removed:
                    # 1. (prev_i, S[0]) -> dist_matrix[tour[prev_i], tour[i]]
                    # 2. (S[-1], next_i) -> dist_matrix[tour[i+length-1], tour[next_i]]
                    # 3. (prev_j, j) -> dist_matrix[tour[prev_j], tour[j]]
                    
                    # Edges added:
                    # 1. (prev_i, next_i) -> dist_matrix[tour[prev_i], tour[next_i]]
                    # 2. (prev_j, S[0]) -> dist_matrix[tour[prev_j], tour[i]]
                    # 3. (S[-1], j) -> dist_matrix[tour[i+length-1], tour[j]]
                    
                    c_old1 = dist_matrix[tour[prev_i], tour[i]]
                    c_old2 = dist_matrix[tour[i + length - 1], tour[next_i]]
                    c_old3 = dist_matrix[tour[prev_j], tour[j]]
                    
                    c_new1 = dist_matrix[tour[prev_i], tour[next_i]]
                    c_new2 = dist_matrix[tour[prev_j], tour[i]]
                    c_new3 = dist_matrix[tour[i + length - 1], tour[j]]
                    
                    delta = (c_new1 + c_new2 + c_new3) - (c_old1 + c_old2 + c_old3)
                    
                    if delta < -1e-9:
                        # Perform the move
                        segment = tour[i : i + length]
                        # Remove segment
                        new_tour = tour[:i] + tour[i + length:]
                        
                        # Determine insertion index in new_tour
                        # If j > i, the index shifts by -length
                        # If j <= i, the index remains j
                        # But we are inserting BEFORE the node that was at j.
                        # In new_tour, the node that was at j is now at:
                        # If j > i: j - length
                        # If j <= i: j
                        # However, we must be careful if j is inside the segment (already handled by continue).
                        
                        if j > i:
                            insert_idx = j - length
                        else:
                            insert_idx = j
                        
                        # Insert segment
                        new_tour[insert_idx:insert_idx] = segment
                        
                        tour = new_tour
                        improved = True
                        break # Restart j loop? Or just continue? 
                        # Restarting i loop is safer to avoid index confusion.
                        # But for speed, we can just break and let the outer while loop restart.
                        # Actually, breaking out of the for j loop and for i loop is best.
                        # Let's break out of all loops.
                        # We can use a flag or raise an exception.
                        # For simplicity, let's just break the innermost and check in the outer.
                        # But we need to restart the search from the beginning to find the next improvement.
                        # So we break out of the for i and for length loops.
                        # Let's use a flag.
                        
            if improved:
                break
        if improved:
            break
    return tour

def _compute_best_tour(n, dist_matrix):
    """Compute a high-quality tour using multi-start NN + 2-Opt + Or-Opt."""
    best_tour = None
    best_len = float('inf')
    
    # Time check
    start_time = time.time()
    
    # 1. Try NN from node 0
    tour = _nearest_neighbor_tour(n, dist_matrix, 0)
    tour = _two_opt(tour, dist_matrix)
    tour = _or_opt(tour, dist_matrix)
    tour_len = _tour_length(tour, dist_matrix)
    if tour_len < best_len:
        best_len = tour_len
        best_tour = tour
    
    # 2. Try NN from other nodes if time permits
    for start in [1, 2, 3]:
        if time.time() - start_time > 10: # Allow 10s for first few
            break
        tour = _nearest_neighbor_tour(n, dist_matrix, start)
        tour = _two_opt(tour, dist_matrix)
        tour = _or_opt(tour, dist_matrix)
        tour_len = _tour_length(tour, dist_matrix)
        if tour_len < best_len:
            best_len = tour_len
            best_tour = tour
    
    # 3. Try random starts if time permits
    num_random_starts = 10
    for _ in range(num_random_starts):
        if time.time() - start_time > 30: # Allow 30s for random starts
            break
        tour = _random_tour(n)
        tour = _two_opt(tour, dist_matrix)
        tour = _or_opt(tour, dist_matrix)
        tour_len = _tour_length(tour, dist_matrix)
        if tour_len < best_len:
            best_len = tour_len
            best_tour = tour
    
    return best_tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in the TSP tour construction.
    Uses a cached, globally optimized tour to guide the constructive step.
    """
    global _cached_dm_fingerprint, _cached_tour
    
    n = distance_matrix.shape[0]
    
    # Check time
    if (time.time() - _START_TIME) > (_DEADLINE - _TIME_MARGIN):
        # Fallback to nearest neighbor if time is running out
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
        
    # Check if we have a cached tour for this instance
    fingerprint = _get_fingerprint(distance_matrix)
    
    if _cached_dm_fingerprint != fingerprint:
        # New instance, compute the optimal tour
        _cached_tour = _compute_best_tour(n, distance_matrix)
        _cached_dm_fingerprint = fingerprint
        
    # Find the next node in the cached tour
    # The tour is a list of node indices.
    # We need to find current_node in the tour and return the next one.
    
    try:
        idx = _cached_tour.index(current_node)
        next_idx = (idx + 1) % n
        next_node = _cached_tour[next_idx]
        
        # Ensure the next node is actually in the unvisited set
        # If not, something is wrong (e.g., cache mismatch), fallback to NN
        if next_node in unvisited_nodes:
            return int(next_node)
        else:
            # Fallback
            return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    except ValueError:
        # current_node not in tour? Fallback
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
# EVOLVE-BLOCK-END
