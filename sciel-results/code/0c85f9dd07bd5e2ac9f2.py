"""Structural change: regret-based constructive TSP heuristic with 1-step lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for time management
_start_time = None
_DEADLINE_SECONDS = 130.0  # Safety margin before 150s limit


def _get_time_remaining():
    """Return seconds remaining before the deadline (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, _DEADLINE_SECONDS - elapsed)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a 1-step lookahead heuristic.
    
    For each candidate c, score = dist(current, c) + min_{c' in unvisited, c' != c} dist(c, c').
    This looks one step ahead: it considers not just how close a node is to current,
    but also how well it connects to the next node we'll likely visit.
    This is structurally different from pure nearest-neighbor or weighted destination heuristics.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n_unvisited = len(unvisited_nodes)
    
    # If only one node left, return it
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If no nodes left (shouldn't happen), return current
    if n_unvisited == 0:
        return int(current_node)
    
    # Time check - if we're running out of time, fall back to nearest neighbor
    remaining_time = _get_time_remaining()
    if remaining_time < 0.5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # For very small sets (<=6), use exact search via brute force on remaining nodes
    if n_unvisited <= 6 and remaining_time > 1.0:
        from itertools import permutations
        best_perm = None
        best_len = float('inf')
        nodes_list = list(unvisited_nodes)
        for perm in permutations(nodes_list):
            # Tour: current -> perm[0] -> ... -> perm[-1] -> destination
            length = distance_matrix[current_node, perm[0]]
            for i in range(len(perm) - 1):
                length += distance_matrix[perm[i], perm[i+1]]
            length += distance_matrix[perm[-1], destination_node]
            if length < best_len:
                best_len = length
                best_perm = perm
        if best_perm is not None:
            return int(best_perm[0])
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # For n_unvisited <= 10, use 2-step lookahead with limited 2-opt
    if n_unvisited <= 10 and remaining_time > 3.0:
        # Build nearest-neighbor tour of remaining nodes
        remaining = list(unvisited_nodes)
        current_pos = current_node
        ordered = []
        while remaining:
            closest = min(remaining, key=lambda x: distance_matrix[current_pos, x])
            ordered.append(closest)
            remaining.remove(closest)
            current_pos = closest
        
        # Apply 2-opt to improve
        full_segment = [current_node] + ordered + [destination_node]
        n = len(full_segment)
        for _ in range(3):
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a, b = full_segment[i], full_segment[i + 1]
                    c, d = full_segment[j], full_segment[(j + 1) % n]
                    old_dist = distance_matrix[a, b] + distance_matrix[c, d]
                    new_dist = distance_matrix[a, c] + distance_matrix[b, d]
                    if new_dist < old_dist - 1e-10:
                        full_segment[i + 1:j + 1] = full_segment[i + 1:j + 1][::-1]
                        improved = True
            if not improved:
                break
        
        if len(full_segment) >= 2:
            return int(full_segment[1])
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # For larger sets: 1-step lookahead
    # For each candidate c, score = dist(current, c) + min_{c' in unvisited, c' != c} dist(c, c')
    # This is O(n^2) but n is at most ~50, so it's fast
    
    # Get distances from current to all unvisited
    curr_to_all = distance_matrix[current_node, unvisited_nodes]
    
    # For each candidate, find the minimum distance to any other unvisited node
    # We need min_{c' != c} dist(c, c') for each c
    # Build sub-matrix of unvisited-to-unvisited distances
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each row i, find the minimum excluding the diagonal (self)
    # Set diagonal to inf, then take min along axis 1
    np.fill_diagonal(sub_dist, np.inf)
    min_other = np.min(sub_dist, axis=1)
    
    # Score: dist(current, c) + min distance from c to any other unvisited
    scores = curr_to_all + min_other
    
    # Also consider connection to destination as a tiebreaker / secondary criterion
    # Blend: primarily lookahead, secondarily destination proximity
    dest_dist = distance_matrix[unvisited_nodes, destination_node]
    max_dest = np.max(dest_dist) if np.max(dest_dist) > 0 else 1.0
    norm_dest = dest_dist / max_dest
    
    # Primary score is the lookahead; add a small penalty for being far from destination
    # This helps guide the tour toward closing it well
    scores = scores + 0.1 * norm_dest * np.max(scores)
    
    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
