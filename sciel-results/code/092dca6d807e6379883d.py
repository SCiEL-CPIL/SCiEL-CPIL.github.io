import itertools
import random
import time

def _is_non_2_colorable(n, edges):
    """
    Fast check for non-2-colorability using bitmask backtracking with unit propagation.
    Returns True if the hypergraph is NOT 2-colorable (i.e., no valid coloring exists).
    """
    if not edges:
        return False
    
    # Precompute edge bitmasks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    full_mask = (1 << n) - 1
    
    # Precompute: for each vertex, which edges contain it (for heuristic)
    # Also precompute edge index by mask for quick lookup? Not needed.
    
    def propagate(color1, unassigned):
        """
        Apply unit propagation. Returns (new_color1, new_unassigned, conflict_flag).
        If conflict, returns (None, None, True).
        """
        changed = True
        while changed:
            changed = False
            for mask in edge_masks:
                assigned_in_edge = mask & ~unassigned
                unassigned_in_edge = mask & unassigned
                
                # Case: all 5 assigned
                if unassigned_in_edge == 0:
                    # Check if monochromatic
                    c1_count = bin(mask & color1).count('1')
                    if c1_count == 5 or c1_count == 0:
                        return None, None, True
                    continue
                
                # Case: exactly 1 unassigned
                if unassigned_in_edge and (unassigned_in_edge & (unassigned_in_edge - 1)) == 0:
                    other_4 = mask ^ unassigned_in_edge
                    # Are all 4 assigned color 1?
                    if (other_4 & ~color1 & full_mask) == 0 and (other_4 & color1) == other_4:
                        # Force to color 0: just remove from unassigned
                        unassigned ^= unassigned_in_edge
                        changed = True
                    # Are all 4 assigned color 0?
                    elif (other_4 & color1) == 0:
                        # Force to color 1
                        color1 |= unassigned_in_edge
                        unassigned ^= unassigned_in_edge
                        changed = True
                        if unassigned == 0:
                            break
        
        return color1, unassigned, False
    
    # Precompute vertex degrees for heuristic
    vertex_degree = [0] * n
    for mask in edge_masks:
        m = mask
        while m:
            v_bit = m & -m
            v_idx = v_bit.bit_length() - 1
            vertex_degree[v_idx] += 1
            m ^= v_bit
    
    def backtrack(color1, unassigned, depth):
        color1, unassigned, conflict = propagate(color1, unassigned)
        if conflict:
            return False
        
        if unassigned == 0:
            # All assigned. Check if any edge is monochromatic.
            for mask in edge_masks:
                c1_count = bin(mask & color1).count('1')
                if c1_count == 5 or c1_count == 0:
                    return False
            return True  # Valid coloring found
        
        # Pick vertex with highest degree among unassigned
        best_v = -1
        best_deg = -1
        m = unassigned
        while m:
            v_bit = m & -m
            v_idx = v_bit.bit_length() - 1
            if vertex_degree[v_idx] > best_deg:
                best_deg = vertex_degree[v_idx]
                best_v = v_bit
            m ^= v_bit
        
        v_bit = best_v
        
        # Try color 1 first (or 0? Let's try both)
        new_color1 = color1 | v_bit
        new_unassigned = unassigned ^ v_bit
        if backtrack(new_color1, new_unassigned, depth + 1):
            return True
            
        # Try color 0
        new_color1 = color1
        new_unassigned = unassigned ^ v_bit
        if backtrack(new_color1, new_unassigned, depth + 1):
            return True
            
        return False

    # Symmetry breaking: Fix vertex 0 to color 0
    initial_color1 = 0
    initial_unassigned = full_mask ^ 1  # Exclude vertex 0
    
    initial_color1, initial_unassigned, conflict = propagate(initial_color1, initial_unassigned)
    if conflict:
        return True
        
    if initial_unassigned == 0:
        for mask in edge_masks:
            c1_count = bin(mask & initial_color1).count('1')
            if c1_count == 5 or c1_count == 0:
                return False
        return False  # Coloring found
        
    if backtrack(initial_color1, initial_unassigned, 0):
        return False  # Coloring found
        
    return True  # No coloring found


def _try_remove_edges(n, edges, deadline):
    """
    Greedily remove edges while maintaining non-2-colorability.
    Returns the pruned edge list.
    """
    current_edges = edges[:]
    removed_any = True
    
    while removed_any and (time.time() < deadline):
        removed_any = False
        random.shuffle(current_edges)
        
        for i in range(len(current_edges) - 1, -1, -1):
            if time.time() > deadline:
                break
                
            e = current_edges[i]
            new_edges = current_edges[:i] + current_edges[i+1:]
            
            if _is_non_2_colorable(n, new_edges):
                current_edges = new_edges
                removed_any = True
    
    return current_edges


def _random_hypergraph(n, num_edges, seed=None):
    """Generate a random 5-uniform hypergraph with num_edges edges."""
    all_combos = list(itertools.combinations(range(n), 5))
    if seed is not None:
        random.seed(seed)
    random.shuffle(all_combos)
    edges = [list(e) for e in all_combos[:num_edges]]
    return edges


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_n = None
    best_edges = None
    best_count = float('inf')
    
    # Strategy 1: Start with K9^(5) and prune
    n = 9
    all_edges_k9 = [list(e) for e in itertools.combinations(range(n), 5)]
    
    # Try multiple random orderings for pruning
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        shuffled = all_edges_k9[:]
        random.shuffle(shuffled)
        
        # Prune with time limit
        sub_deadline = min(deadline, start_time + 20 + attempt * 5)
        pruned = _try_remove_edges(n, shuffled, sub_deadline)
        
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_n = n
            best_edges = pruned[:]
    
    # Strategy 2: Try n=10, 11 with random sparse hypergraphs
    for n_try in [10, 11, 12]:
        if time.time() > deadline:
            break
        
        # Try different edge counts
        for num_e in range(30, 80, 5):
            if time.time() > deadline:
                break
            
            for seed in range(3):
                if time.time() > deadline:
                    break
                
                edges = _random_hypergraph(n_try, num_e, seed=seed + n_try * 100)
                
                # Check if non-2-colorable
                if _is_non_2_colorable(n_try, edges):
                    # Prune
                    sub_deadline = min(deadline, time.time() + 10)
                    pruned = _try_remove_edges(n_try, edges, sub_deadline)
                    
                    if len(pruned) < best_count:
                        best_count = len(pruned)
                        best_n = n_try
                        best_edges = pruned[:]
    
    # Strategy 3: If we have a good result, try more aggressive pruning
    if best_edges is not None and best_count > 40:
        if time.time() < deadline - 10:
            # Try removing edges in different orders
            for attempt in range(3):
                if time.time() > deadline:
                    break
                
                shuffled = best_edges[:]
                random.shuffle(shuffled)
                sub_deadline = min(deadline, time.time() + 15)
                pruned = _try_remove_edges(best_n, shuffled, sub_deadline)
                
                if len(pruned) < best_count:
                    best_count = len(pruned)
                    best_edges = pruned[:]
    
    # Final verification
    if best_edges is None or not _is_non_2_colorable(best_n, best_edges):
        # Fallback to K9
        return {'n': 9, 'edges': all_edges_k9}
    
    return {'n': best_n, 'edges': best_edges}
