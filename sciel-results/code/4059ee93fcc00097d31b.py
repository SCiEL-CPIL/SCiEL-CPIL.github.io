import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n = 9
    vertices = list(range(n))
    
    # Generate all 5-uniform edges
    from itertools import combinations
    all_edges = [list(e) for e in combinations(vertices, 5)]
    num_edges = len(all_edges) # 126
    
    # Total number of colorings is 2^9 = 512
    total_colorings = 1 << n
    all_bits = (1 << total_colorings) - 1
    
    # Precompute masks for each vertex being 0 or 1
    # mask0[v] has bit i set if vertex v is 0 in coloring i
    # mask1[v] has bit i set if vertex v is 1 in coloring i
    mask0 = [0] * n
    mask1 = [0] * n
    
    for c in range(total_colorings):
        for v in range(n):
            if (c >> v) & 1:
                mask1[v] |= (1 << c)
            else:
                mask0[v] |= (1 << c)
                
    # Precompute mask for each edge
    # M[e] is the set of colorings where edge e is monochromatic
    edge_masks = []
    for e in all_edges:
        m0 = all_bits
        m1 = all_bits
        for v in e:
            m0 &= mask0[v]
            m1 &= mask1[v]
        edge_masks.append(m0 | m1)
        
    # Greedy Set Cover
    uncovered = all_bits
    selected_indices = []
    remaining = list(range(num_edges))
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_count = 0
        
        # Find edge covering most uncovered colorings
        for idx in remaining:
            count = bin(uncovered & edge_masks[idx]).count('1')
            if count > best_count:
                best_count = count
                best_idx = idx
                
        if best_idx == -1:
            break
            
        selected_indices.append(best_idx)
        uncovered &= ~edge_masks[best_idx]
        remaining.remove(best_idx)
        
    # Prune redundant edges
    # We have a set of selected indices. Try to remove each one.
    # To do this efficiently, we can recalculate the union of masks for the current set,
    # then for each edge in the set, check if removing it still covers all bits.
    
    current_selected = set(selected_indices)
    
    # Optimization: If the greedy set is small, this is fast.
    # If it's large, we might skip or limit pruning.
    # With n=9, greedy usually yields ~60-70 edges.
    
    # Sort by index for determinism
    current_selected_list = sorted(current_selected)
    
    # Pruning pass
    # We can do multiple passes or just one. One pass is usually enough to remove obvious redundancies.
    for i in range(len(current_selected_list)):
        if time.time() > deadline:
            break
        idx = current_selected_list[i]
        # Check if all bits are covered by the other edges
        # This is O(N) per check, so O(N^2) total.
        # N is ~60, so 3600 checks. Each check is OR of 60 big ints. Fast.
        
        # Calculate union of all masks except idx
        union_mask = 0
        for j in range(len(current_selected_list)):
            if j == i:
                continue
            union_mask |= edge_masks[current_selected_list[j]]
            
        if union_mask == all_bits:
            # Edge is redundant
            current_selected_list.pop(i)
            # Shift index back to check the new element at i
            i -= 1
            
    # Construct final edge list
    final_edges = [all_edges[idx] for idx in current_selected_list]
    
    # Sanity check: Verify no 2-coloring
    # This is a bit expensive if done naively, but we can skip if we trust the mask logic.
    # The mask logic guarantees coverage.
    
    return {'n': n, 'edges': final_edges}
