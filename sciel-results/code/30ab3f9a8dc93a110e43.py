import time
import random
from itertools import combinations

def _is_2_colorable_fast(n, edges):
    """
    Fast check for 2-colorability using backtracking with unit propagation.
    Returns True if 2-colorable, False if not.
    """
    if not edges:
        return True
        
    m = len(edges)
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        
    all_vars = (1 << n) - 1
    
    import sys
    sys.setrecursionlimit(10000)

    def backtrack(assign, unassigned):
        changed = True
        while changed:
            changed = False
            for mask in edge_masks:
                assigned_in_edge = mask & ~unassigned
                unassigned_in_edge = mask & unassigned
                
                if unassigned_in_edge == 0:
                    if (assign & mask) == mask or (assign & mask) == 0:
                        return None
                else:
                    assigned_zero = assigned_in_edge & ~assign
                    assigned_one = assigned_in_edge & assign
                    unassigned_count = bin(unassigned_in_edge).count('1')
                    
                    if assigned_one == 0 and unassigned_count == 1:
                        v = unassigned_in_edge.bit_length() - 1
                        assign |= (1 << v)
                        unassigned &= ~(1 << v)
                        changed = True
                        continue
                    if assigned_zero == 0 and unassigned_count == 1:
                        v = unassigned_in_edge.bit_length() - 1
                        assign &= ~(1 << v)
                        unassigned &= ~(1 << v)
                        changed = True
                        continue
                        
        if unassigned == 0:
            for mask in edge_masks:
                if (assign & mask) == mask or (assign & mask) == 0:
                    return None
            return assign
            
        v = unassigned.bit_length() - 1
        res = backtrack(assign | (1 << v), unassigned & ~(1 << v))
        if res is not None:
            return res
        return backtrack(assign & ~(1 << v), unassigned & ~(1 << v))
        
    return backtrack(0, all_vars) is not None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s buffer for overhead
    
    n = 11
    # All possible 5-edges on 11 vertices
    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)
    
    # Precompute masks for Set Cover
    # Universe size is 2^11 = 2048
    universe_size = 1 << n
    full_universe_mask = (1 << universe_size) - 1
    
    # edge_mask[i] is a bitmask of length 2048, where bit j is 1 if edge i is monochromatic in coloring j
    edge_masks = [0] * num_edges
    for i, e in enumerate(all_edges):
        mask_val = 0
        for coloring in range(universe_size):
            # Check if coloring makes edge e monochromatic
            # Get colors of vertices in e
            colors = set()
            for v in e:
                colors.add((coloring >> v) & 1)
            if len(colors) == 1:
                mask_val |= (1 << coloring)
        edge_masks[i] = mask_val
        
    # Greedy Set Cover
    uncovered = full_universe_mask
    selected_indices = []
    
    while uncovered != 0:
        best_idx = -1
        best_gain = 0
        for i in range(num_edges):
            if i in selected_indices:
                continue
            gain = bin(edge_masks[i] & uncovered).count('1')
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1:
            break # Should not happen
        selected_indices.append(best_idx)
        uncovered &= ~edge_masks[best_idx]
        
    # Local Search: Try to remove edges
    # We want to minimize the number of edges.
    # If removing an edge leaves the hypergraph non-2-colorable, remove it.
    changed = True
    while changed:
        changed = False
        for i in range(len(selected_indices) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = selected_indices[:i] + selected_indices[i+1:]
            # Check if candidate is still non-2-colorable
            edges_to_check = [all_edges[idx] for idx in candidate]
            if not _is_2_colorable_fast(n, edges_to_check):
                # It is still non-2-colorable, so we can remove this edge
                selected_indices = candidate
                changed = True
                break # Restart the loop from the end
        
        if time.time() > deadline:
            break

    # If we still have too many edges or time is running out, we might need a fallback
    # But greedy+local search on this instance is very fast.
    
    final_edges = [list(all_edges[idx]) for idx in selected_indices]
    
    # Fallback if something went wrong (e.g. empty set or timeout)
    if not final_edges or time.time() > deadline:
         # Complete 5-graph on 9 vertices is definitely non-2-colorable
         fallback_edges = [list(e) for e in combinations(range(9), 5)]
         return {'n': 9, 'edges': fallback_edges}
         
    return {'n': n, 'edges': final_edges}
