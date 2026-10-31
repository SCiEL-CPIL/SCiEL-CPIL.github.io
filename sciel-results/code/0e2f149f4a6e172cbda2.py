"""Traveling salesman constructive heuristic with global 2-Opt improvement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Build a fast initial tour using a hybrid constructive heuristic,
# then spend the remaining time budget on global 2-Opt improvements.
#
# The constructive phase uses a combination of:
#   - Nearest neighbor for the first few steps (fast, good local structure)
#   - A lookahead-2 + connectivity penalty for the rest (balances local and global)
#
# The improvement phase repeatedly applies 2-Opt moves (reversing a segment
# of the tour) to reduce total tour length. We use a simple O(n^2) 2-Opt
# per pass, and run as many passes as the time budget allows.
#
# This is structurally different from the previous approach because:
#   1. It doesn't just greedily pick the next node; it builds a full tour first.
#   2. It then globally optimizes the tour structure via 2-Opt, which can
#      escape local optima of the greedy construction.
#   3. The time budget is explicitly managed: ~2s for construction, rest for 2-Opt.

def _construct_initial_tour(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Build an initial tour using a hybrid greedy heuristic."""
    n = distance_matrix.shape[0]
    visited = [current_node]
    unvisited = set(unvisited_nodes.tolist())
    
    t_start = time.time()
    CONSTRUCT_BUDGET = 2.0  # seconds for construction phase
    
    while unvisited:
        # Check time budget
        if time.time() - t_start > CONSTRUCT_BUDGET:
            # Fallback: just append remaining nodes in order of nearest neighbor
            while unvisited:
                candidates = list(unvisited)
                dists = distance_matrix[visited[-1], candidates]
                best_idx = np.argmin(dists)
                next_node = candidates[best_idx]
                visited.append(next_node)
                unvisited.remove(next_node)
            break
        
        if len(unvisited) == 1:
            next_node = unvisited.pop()
            visited.append(next_node)
            break
        
        # Convert to array for numpy operations
        unvisited_arr = np.array(list(unvisited))
        
        # Distance from current to each candidate
        dists_from_current = distance_matrix[visited[-1], unvisited_arr]
        
        # Sub-matrix of distances between unvisited nodes
        sub_matrix = distance_matrix[np.ix_(unvisited_arr, unvisited_arr)]
        sub_matrix_no_diag = sub_matrix.copy()
        np.fill_diagonal(sub_matrix_no_diag, np.inf)
        
        # Lookahead: min distance from each candidate to any other unvisited node
        min_dist_to_other = np.min(sub_matrix_no_diag, axis=1)
        
        # Connectivity penalty: mean distance from each candidate to all other unvisited nodes
        sum_dist_to_others = np.sum(sub_matrix_no_diag, axis=1)
        mean_dist_to_other = sum_dist_to_others / max(len(unvisited_arr) - 1, 1)
        
        # Combine scores
        w1 = 0.5  # Reduced weight for lookahead to be less aggressive
        w2 = 0.2  # Reduced weight for connectivity
        scores = dists_from_current + w1 * min_dist_to_other + w2 * mean_dist_to_other
        
        best_idx = np.argmin(scores)
        next_node = int(unvisited_arr[best_idx])
        visited.append(next_node)
        unvisited.remove(next_node)
    
    # The tour is a closed loop, so we don't need to explicitly add the return to start
    # The tour is: visited[0] -> visited[1] -> ... -> visited[-1] -> visited[0]
    return visited


def _tour_length(tour, distance_matrix):
    """Compute the total length of a closed tour."""
    n = len(tour)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        total += distance_matrix[u, v]
    return total


def _two_opt_improve(tour, distance_matrix, time_budget):
    """Apply 2-Opt improvements to the tour until time budget is exhausted."""
    n = len(tour)
    if n < 4:
        return tour
    
    tour = list(tour)
    current_length = _tour_length(tour, distance_matrix)
    
    t_start = time.time()
    DEADLINE = t_start + time_budget
    
    improved = True
    while improved:
        improved = False
        # Check time budget
        if time.time() > DEADLINE:
            break
        
        # Try all 2-Opt moves
        for i in range(n - 2):
            if time.time() > DEADLINE:
                break
            
            for j in range(i + 2, n):
                if time.time() > DEADLINE:
                    break
                
                # Skip if i=0 and j=n-1 (that's the whole tour reversal, no change)
                if i == 0 and j == n - 1:
                    continue
                
                # Calculate the change in length if we reverse tour[i+1..j]
                a = tour[i]
                b = tour[i + 1]
                c = tour[j]
                d = tour[(j + 1) % n]
                
                # Current edges: (a,b) and (c,d)
                # New edges: (a,c) and (b,d)
                delta = (distance_matrix[a, c] + distance_matrix[b, d]) - \
                        (distance_matrix[a, b] + distance_matrix[c, d])
                
                if delta < -1e-10:
                    # Apply the move
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    current_length += delta
                    improved = True
    
    return tour


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node by building a full tour and then optimizing it with 2-Opt.
    
    Note: This function is called iteratively by the evaluator to build the tour.
    However, the 2-Opt improvement requires a complete tour. So we need to be
    strategic: we'll build a fast initial tour in the first call, then use
    2-Opt on subsequent calls to guide the selection.
    
    Actually, looking at the interface more carefully, select_next_node is called
    once per step to pick the next city. So we can't do 2-Opt on the full tour
    until the tour is complete. 
    
    Revised strategy: Use a fast constructive heuristic for the selection, but
    with a smarter scoring function. We'll use a combination of:
    1. Immediate distance
    2. Lookahead (best next step from candidate)
    3. A "potential" term based on the candidate's position relative to the
       overall point cloud (e.g., distance to centroid, or angle-based)
    
    For now, let's use a simpler but effective approach: nearest neighbor with
    a small penalty for nodes that are too far from the current "center of mass"
    of unvisited nodes. This encourages a more balanced tour.
    """
    if len(unvisited_nodes) == 0:
        return -1
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_remaining = len(unvisited_nodes)
    t_start = time.time()
    DEADLINE = 145.0
    
    # 1. Distance from current to each candidate
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Sub-matrix of distances between unvisited nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    sub_matrix_no_diag = sub_matrix.copy()
    np.fill_diagonal(sub_matrix_no_diag, np.inf)
    
    # 3. Lookahead: min distance from each candidate to any other unvisited node
    min_dist_to_other = np.min(sub_matrix_no_diag, axis=1)
    
    # 4. Connectivity penalty: mean distance from each candidate to all other unvisited nodes
    sum_dist_to_others = np.sum(sub_matrix_no_diag, axis=1)
    mean_dist_to_other = sum_dist_to_others / max(n_remaining - 1, 1)
    
    # 5. Combine scores with tuned weights
    w1 = 0.3
    w2 = 0.1
    scores = dists_from_current + w1 * min_dist_to_other + w2 * mean_dist_to_other
    
    # Check deadline
    if time.time() > DEADLINE:
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
