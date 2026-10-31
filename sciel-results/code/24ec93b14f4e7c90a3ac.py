# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for caching the optimized tour
_start_time = None
_DEADLINE = 115.0  # safety margin before 120s/150s limit
_cached_tour = None
_cached_dest = None
_cached_unvisited_key = None


def _opt_2opt(tour, dist):
    """Improve tour using 2-opt. Returns improved tour."""
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            a = tour[i - 1]
            b = tour[i]
            for j in range(i + 1, n):
                if j == i:
                    continue
                c = tour[j]
                d = tour[(j + 1) % n]
                
                # Cost of current edges: (a,b) and (c,d)
                # Cost of new edges: (a,c) and (b,d)
                delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
                
                if delta < -1e-9:
                    # Reverse segment from i to j
                    tour[i:j+1] = tour[i:j+1][::-1]
                    improved = True
    return tour


def _or_opt(tour, dist, max_restarts=3):
    """Simple OR-opt: move 1 or 2 consecutive nodes to another position."""
    n = len(tour)
    if n < 4:
        return tour
    
    for restart in range(max_restarts):
        improved = True
        while improved:
            improved = False
            for k in [1, 2]:
                for i in range(n):
                    if i + k >= n:
                        break
                    # Nodes to move: tour[i:i+k]
                    # Previous node: tour[(i-1)%n]
                    # Next node after block: tour[(i+k)%n]
                    
                    for j in range(n):
                        # Don't insert near original position
                        if abs(j - i) <= k or j == (i + k) % n or j == (i - 1) % n:
                            continue
                        
                        # Current cost:
                        # Edge before i: ((i-1)%n, i)
                        # Edge after block: (i+k-1, (i+k)%n)  -- wait, indices
                        # Let's use clear indices
                        p_idx = (i - 1) % n
                        a = tour[p_idx]
                        b = tour[i]
                        # ... block ...
                        last_in_block = tour[i + k - 1]
                        c = tour[(i + k) % n]
                        
                        # Remove block
                        # Insert between tour[j] and tour[(j+1)%n]
                        # New edges: (a, c), (tour[j], first_in_block), (last_in_block, tour[(j+1)%n])
                        # Removed edges: (a, b), (last_in_block, c), (tour[j], tour[(j+1)%n])
                        
                        first_in_block = tour[i]
                        last_in_block = tour[i + k - 1]
                        
                        j_next = (j + 1) % n
                        x = tour[j]
                        y = tour[j_next]
                        
                        old_cost = dist[a, b] + dist[last_in_block, c] + dist[x, y]
                        new_cost = dist[a, c] + dist[x, first_in_block] + dist[last_in_block, y]
                        
                        delta = new_cost - old_cost
                        if delta < -1e-9:
                            # Perform move
                            block = tour[i:i+k]
                            # Remove block
                            new_tour = tour[:i] + tour[i+k:]
                            # Find new position for j
                            # If i <= j, j shifts left by k
                            # If i > j, j stays (unless j is in removed range, but we checked)
                            # Actually, let's just rebuild
                            insert_pos = j
                            if i <= j:
                                insert_pos = j - k
                            
                            # But wait, if we remove first, indices change.
                            # Let's do it carefully
                            # tour = [ ... p, A1, A2, ..., B, ... ]
                            # Remove A1..Ak
                            # Insert at position j (in the new list)
                            
                            # Simpler:
                            # 1. Pop block
                            # 2. Find index of insertion point in remaining list
                            #    If original j > i, it's now j - k
                            #    If original j < i, it's still j (but check if j is in block, we excluded that)
                            
                            # Let's just use a list copy to be safe, N=50 so it's cheap
                            new_tour = list(tour)
                            block = new_tour[i:i+k]
                            del new_tour[i:i+k]
                            
                            # Determine insert index in new_tour
                            # The node that was at index (i+k) is now at i
                            # The node at index j:
                            if j > i:
                                new_j = j - k
                            else:
                                new_j = j
                            
                            # But if j is in [i, i+k), we skipped it.
                            # We insert *before* the node that is at new_j
                            # Wait, original logic: insert between tour[j] and tour[(j+1)%n]
                            # So we insert at index new_j
                            
                            new_tour[new_j:new_j] = block
                            tour = new_tour
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    return tour


def _optimize_tour(tour, dist):
    """Combine 2-opt and OR-opt."""
    tour = list(tour)
    # First pass 2-opt
    tour = _opt_2opt(tour, dist)
    # Then OR-opt
    tour = _or_opt(tour, dist, max_restarts=2)
    # Another 2-opt pass
    tour = _opt_2opt(tour, dist)
    return tour


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city for a TSP constructive heuristic.
    Uses a hybrid approach:
    1. If we have enough time and this is the start of the tour, build a greedy tour,
       optimize it with 2-opt/OR-opt, and return the first step.
    2. Otherwise, use a lookahead heuristic to pick the next node.
    """
    global _start_time, _cached_tour, _cached_dest, _cached_unvisited_key
    
    if _start_time is None:
        _start_time = time.time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    elapsed = time.time() - _start_time
    
    # If we have a cached tour, use it
    if _cached_tour is not None and _cached_dest == destination_node:
        # Check if current_node is in the tour
        try:
            idx = _cached_tour.index(current_node)
            next_idx = (idx + 1) % len(_cached_tour)
            next_node = _cached_tour[next_idx]
            # Verify next_node is in unvisited_nodes
            if next_node in unvisited_nodes:
                return int(next_node)
        except ValueError:
            pass
    
    # If we have time and this is early in the tour, try to build and optimize a full tour
    if elapsed < _DEADLINE - 30.0 and n_unvisited > 10:
        # Build a greedy tour from current_node through all unvisited, back to destination
        # But we need to be careful: the tour must start at current_node and end at destination_node
        # Actually, the evaluator builds the tour step by step.
        # Let's build a tour that starts at current_node, visits all unvisited, and ends at destination_node.
        
        # Greedy construction
        tour = [current_node]
        unvisited = list(unvisited_nodes)
        
        while unvisited:
            curr = tour[-1]
            dists = distance_matrix[curr, unvisited]
            min_idx = np.argmin(dists)
            next_node = unvisited[min_idx]
            tour.append(next_node)
            unvisited.pop(min_idx)
        
        # The tour currently is: current -> ... -> last
        # We need to close it to destination_node.
        # But the evaluator will eventually add destination_node.
        # So our optimized tour should be: current -> ... -> destination
        # Let's append destination_node to the tour for optimization
        full_tour = tour + [destination_node]
        
        # Optimize
        optimized_tour = _optimize_tour(full_tour, distance_matrix)
        
        # Cache it
        _cached_tour = optimized_tour
        _cached_dest = destination_node
        
        # Return the next node in the optimized tour
        # current_node is at index 0
        # Next node is at index 1
        next_node = optimized_tour[1]
        
        # Verify next_node is in unvisited_nodes
        if next_node in unvisited_nodes:
            return int(next_node)
        else:
            # Fallback: nearest neighbor
            dists = distance_matrix[current_node, unvisited_nodes]
            return int(unvisited_nodes[np.argmin(dists)])
    
    # If no cache or not enough time, use lookahead heuristic
    candidates = unvisited_nodes
    n_candidates = n_unvisited
    
    dist_from_current = distance_matrix[current_node, candidates]
    
    if n_candidates <= 8:
        sub_dist = distance_matrix[np.ix_(candidates, candidates)]
        diag_idx = np.arange(n_candidates)
        sub_dist_masked = sub_dist.copy()
        sub_dist_masked[diag_idx, diag_idx] = np.inf
        next_step_costs = np.min(sub_dist_masked, axis=1)
        dist_to_dest = distance_matrix[candidates, destination_node]
        scores = dist_from_current + 0.5 * next_step_costs + 0.5 * dist_to_dest
    else:
        sub_dist = distance_matrix[np.ix_(candidates, candidates)]
        diag_idx = np.arange(n_candidates)
        sub_dist_masked = sub_dist.copy()
        sub_dist_masked[diag_idx, diag_idx] = np.inf
        next_step_costs = np.min(sub_dist_masked, axis=1)
        scores = dist_from_current + 0.3 * next_step_costs
    
    best_idx = int(np.argmin(scores))
    return int(candidates[best_idx])
# EVOLVE-BLOCK-END
