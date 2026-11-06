"""
TSP Constructive Heuristic: Regret-Greedy with Selective 2-Opt Refinement
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track the tour and time
_current_tour = []
_start_time = None
_TIME_LIMIT = 145.0  # Safety margin for 150s limit

def _reset_state():
    global _current_tour, _start_time
    _current_tour = []
    _start_time = time.time()

def _greedy_completion(start_node, unvisited_nodes, distance_matrix, end_node):
    """
    Fast greedy nearest-neighbor completion.
    Returns the total distance of the path from start_node through all unvisited_nodes to end_node.
    """
    if len(unvisited_nodes) == 0:
        return distance_matrix[start_node, end_node]
    
    remaining = list(unvisited_nodes)
    curr = start_node
    total_dist = 0.0
    
    # Optimization: Pre-allocate or use efficient list operations
    # For N=50, this loop is small.
    while remaining:
        # Find nearest neighbor in remaining
        # Vectorized min
        dists = distance_matrix[curr, remaining]
        min_idx = np.argmin(dists)
        next_node = remaining[min_idx]
        total_dist += dists[min_idx]
        curr = next_node
        remaining.pop(min_idx)
        
    total_dist += distance_matrix[curr, end_node]
    return total_dist

def _fast_2opt_single_pass(tour, distance_matrix):
    """
    Perform a single pass of 2-opt improvement on a closed tour.
    Returns the improved tour length.
    This is a lightweight refinement step.
    """
    n = len(tour)
    if n < 4:
        return sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    
    t = tour[:]
    improved = True
    # Limit passes to 1 to keep it fast, or until no improvement in this pass
    # We just want to see if a simple reversal helps.
    
    # Single pass over all i, j
    for i in range(n - 1):
        # j must be at least i+2
        # Vectorized check for best j for this i? 
        # For simplicity and speed in Python, we check a few promising j's or just do a linear scan if N is small.
        # N=50, so n^2/2 = 1250 checks. This is fast.
        best_j = -1
        best_gain = 0
        d_i_i1 = distance_matrix[t[i], t[i+1]]
        
        for j in range(i+2, n):
            # Skip if j is adjacent to i in the cycle (handled by i+2 start and n-1 end logic? No, j goes to n-1)
            # If j == n-1, next is 0.
            next_j = 0 if j == n-1 else j+1
            
            # Old edges: (i, i+1) and (j, next_j)
            # New edges: (i, j) and (i+1, next_j)
            old_cost = d_i_i1 + distance_matrix[t[j], t[next_j]]
            new_cost = distance_matrix[t[i], t[j]] + distance_matrix[t[i+1], t[next_j]]
            
            gain = old_cost - new_cost
            if gain > best_gain:
                best_gain = gain
                best_j = j
                
        if best_j != -1 and best_gain > 1e-8:
            # Reverse segment i+1 to best_j
            t[i+1:best_j+1] = t[i+1:best_j+1][::-1]
            # After reversal, we could restart, but for a single pass estimate, we continue or break.
            # Breaking and restarting is better for quality, but we are in a "single pass" heuristic mode for speed.
            # Let's just apply the best move found in this i-loop iteration and continue to next i.
            # Actually, applying a move changes the tour, so subsequent i checks are on the new tour.
            pass
            
    # Calculate final length
    length = 0.0
    for i in range(n):
        length += distance_matrix[t[i], t[(i+1)%n]]
    return length

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _current_tour, _start_time
    
    # Initialize on first call
    if _start_time is None:
        _start_time = time.time()
        _current_tour = [current_node]
        
    # The destination_node in TSP is typically the start node to close the loop.
    # However, the interface provides destination_node. In many constructive TSP contexts,
    # if it's a cycle, destination is the start. If it's a path, it's the end.
    # The prompt says "closed-tour length", so destination_node is likely the start node.
    # We will assume the tour closes back to the first node in _current_tour.
    start_node = _current_tour[0]
    
    # Time check
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Fallback to simple nearest neighbor to finish quickly
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if len(unvisited_nodes) == 0:
        return -1
    
    if len(unvisited_nodes) == 1:
        next_node = int(unvisited_nodes[0])
        _current_tour.append(next_node)
        return next_node
    
    if len(unvisited_nodes) == 2:
        a, b = unvisited_nodes[0], unvisited_nodes[1]
        cost1 = distance_matrix[current_node, a] + distance_matrix[a, b] + distance_matrix[b, start_node]
        cost2 = distance_matrix[current_node, b] + distance_matrix[b, a] + distance_matrix[a, start_node]
        if cost1 <= cost2:
            next_node = int(a)
        else:
            next_node = int(b)
        _current_tour.append(next_node)
        return next_node
    
    n_cands = len(unvisited_nodes)
    
    # 1. Calculate immediate cost and greedy future cost for all candidates
    imm_costs = distance_matrix[current_node, unvisited_nodes]
    future_costs = np.zeros(n_cands)
    
    # To speed up, we can parallelize or just loop. N=50 is small.
    for i in range(n_cands):
        cand = unvisited_nodes[i]
        # Remove cand from unvisited for the completion estimate
        # Creating a new list is O(N). 50 * 50 = 2500 ops. Fine.
        remaining_mask = np.ones(n_cands, dtype=bool)
        remaining_mask[i] = False
        remaining = unvisited_nodes[remaining_mask]
        
        future_costs[i] = _greedy_completion(cand, remaining, distance_matrix, start_node)
        
    total_scores = imm_costs + future_costs
    
    # 2. Select Top K candidates for refinement
    # Sort indices by score
    sorted_indices = np.argsort(total_scores)
    top_k = min(3, n_cands)
    top_candidates_indices = sorted_indices[:top_k]
    
    # 3. Refine: For each top candidate, simulate the tour and apply a light 2-opt
    best_candidate_node = int(unvisited_nodes[sorted_indices[0]]) # Default to best greedy
    best_refined_length = float('inf')
    
    for idx in top_candidates_indices:
        cand = unvisited_nodes[idx]
        # Construct the tentative tour: _current_tour + [cand] + greedy_completion(cand)
        # We need the actual nodes from the greedy completion to apply 2-opt.
        # _greedy_completion currently only returns length. We need the path.
        
        # Let's modify the approach: Re-run greedy to get the path for the top candidates only.
        # This is expensive if we do it for all, but only for top 3 it's fine.
        
        remaining_mask = np.ones(n_cands, dtype=bool)
        remaining_mask[idx] = False
        remaining = unvisited_nodes[remaining_mask]
        
        # Get path
        path = [cand]
        curr = cand
        rem_list = list(remaining)
        while rem_list:
            dists = distance_matrix[curr, rem_list]
            min_idx = np.argmin(dists)
            next_node = rem_list[min_idx]
            path.append(next_node)
            curr = next_node
            rem_list.pop(min_idx)
        
        full_tour = _current_tour + path + [start_node]
        
        # Apply fast 2-opt single pass
        refined_length = _fast_2opt_single_pass(full_tour, distance_matrix)
        
        if refined_length < best_refined_length:
            best_refined_length = refined_length
            best_candidate_node = int(cand)
            
    _current_tour.append(best_candidate_node)
    
    return best_candidate_node
# EVOLVE-BLOCK-END
