import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s buffer for the 150s limit
    
    best_edges = None
    best_n = 0
    min_edges_found = float('inf')

    # We try a range of n. 
    # Known results suggest n around 10-14 might be optimal for 51 edges.
    # n=9 has 126 edges total. n=10 has 252. n=11 has 462.
    # The set cover problem size is 2^n. 
    # n=10 -> 1024 colorings.
    # n=11 -> 2048 colorings.
    # n=12 -> 4096 colorings.
    # n=13 -> 8192 colorings.
    # n=14 -> 16384 colorings.
    
    # For each n, we run greedy set cover and then reduction.
    
    for n in range(9, 15):
        if time.time() > deadline - 10.0:
            break
            
        try:
            # 1. Generate all 5-uniform edges
            all_edges = list(combinations(range(n), 5))
            num_edges = len(all_edges)
            
            # 2. Precompute coverage masks for each edge
            # A coloring is an integer c from 0 to 2^n - 1.
            # Bit v of c indicates color of vertex v.
            # Edge e is monochromatic in coloring c if:
            # (c & mask_e) == mask_e (all 1s) OR (c & mask_e) == 0 (all 0s)
            
            # We represent the set of colorings covered by an edge as a Python int bitmask.
            # Bit i of this mask is 1 if coloring i is covered.
            # Total colorings = 2^n. If 2^n > 64, Python ints handle it fine.
            
            total_colorings = 1 << n
            full_mask = (1 << total_colorings) - 1
            
            edge_cover_masks = []
            
            # Optimization: 
            # For a given edge e, the colorings that make it monochromatic are those 
            # where all vertices in e are 0 or all are 1.
            # We can construct these masks more efficiently than iterating all 2^n colorings.
            # However, for n up to 14, 2^14 = 16384. 
            # Iterating 16384 colorings for each of ~1000 edges is 16M ops. 
            # Python loops are slow. 16M iterations might take ~1-2 seconds.
            # Let's try to optimize the mask generation.
            
            # Alternative: Iterate over colorings once and update masks for all edges?
            # No, that's 2^n * num_edges. Same complexity.
            
            # Let's stick to per-edge generation but optimize the inner loop.
            # For edge e, let mask_e be the bitmask of vertices in e.
            # We want to set bit c in edge_cover_masks[i] if (c & mask_e) in {0, mask_e}.
            
            # Actually, we can generate the set of "monochromatic colorings" for an edge directly.
            # If edge vertices are fixed to 0, the other n-5 vertices can be anything.
            # If edge vertices are fixed to 1, the other n-5 vertices can be anything.
            
            # Let's precompute the list of "free" vertices for each edge? 
            # Or just iterate.
            
            # Let's use a helper to generate the mask.
            # For n=14, 16384 colorings. 
            # num_edges for n=14 is C(14,5) = 2002.
            # 2002 * 16384 ~ 32M operations. This might be too slow in pure Python if not careful.
            
            # Optimization: 
            # Instead of iterating all colorings, we can construct the mask.
            # The set of colorings where edge e is all 0s is:
            # { c | c & mask_e == 0 }
            # The set of colorings where edge e is all 1s is:
            # { c | c & mask_e == mask_e }
            
            # We can iterate over the submasks of the complement of mask_e?
            # Let comp_mask = full_vertex_mask ^ mask_e.
            # Colorings with all 0s on e: c = s, where s is a submask of comp_mask.
            # Colorings with all 1s on e: c = mask_e | s, where s is a submask of comp_mask.
            
            # Number of submasks of comp_mask is 2^(n-5).
            # For n=14, 2^9 = 512.
            # So for each edge, we iterate 512 submasks for 0s and 512 for 1s. Total 1024 ops per edge.
            # 2002 edges * 1024 ops ~ 2M ops. This is much faster than 32M.
            
            full_vertex_mask = (1 << n) - 1
            
            for e in all_edges:
                mask_e = 0
                for v in e:
                    mask_e |= (1 << v)
                
                comp_mask = full_vertex_mask ^ mask_e
                
                cover_mask = 0
                
                # Iterate submasks of comp_mask
                # s = 0; while s <= comp_mask: ... s = (s+1) & comp_mask
                # Standard submask iteration:
                s = comp_mask
                while True:
                    # Case 1: Edge vertices are 0. Coloring is s.
                    cover_mask |= (1 << s)
                    
                    # Case 2: Edge vertices are 1. Coloring is mask_e | s.
                    cover_mask |= (1 << (mask_e | s))
                    
                    if s == 0:
                        break
                    s = (s - 1) & comp_mask
                
                edge_cover_masks.append(cover_mask)
            
            # 3. Greedy Set Cover
            # We want to select edges such that the union of their cover masks is full_mask.
            # uncovered = full_mask
            uncovered = full_mask
            selected_indices = []
            used = [False] * num_edges
            
            while uncovered:
                if time.time() > deadline - 5.0:
                    break
                
                best_idx = -1
                best_count = -1
                
                for i in range(num_edges):
                    if used[i]:
                        continue
                    # Count how many bits in edge_cover_masks[i] are set in uncovered
                    # new_covered = edge_cover_masks[i] & uncovered
                    # count = new_covered.bit_count()
                    new_covered = edge_cover_masks[i] & uncovered
                    count = new_covered.bit_count()
                    
                    if count > best_count:
                        best_count = count
                        best_idx = i
                
                if best_idx == -1:
                    # Should not happen if we started with full coverage potential
                    break
                
                selected_indices.append(best_idx)
                used[best_idx] = True
                uncovered &= ~edge_cover_masks[best_idx]
            
            # 4. Reduction and Local Search Phase
            current_selected = selected_indices[:]
            
            # Helper to check if a set of indices covers full_mask
            def covers_full(indices):
                union = 0
                for idx in indices:
                    union |= edge_cover_masks[idx]
                return union == full_mask
            
            # First, simple reduction: remove redundant edges
            changed = True
            while changed:
                changed = False
                if time.time() > deadline - 5.0:
                    break
                for i in range(len(current_selected)):
                    idx_to_remove = current_selected[i]
                    # Check if removing idx_to_remove still covers everything
                    union_mask = 0
                    for j in range(len(current_selected)):
                        if j != i:
                            union_mask |= edge_cover_masks[current_selected[j]]
                    
                    if union_mask == full_mask:
                        current_selected.pop(i)
                        changed = True
                        break # Restart loop
            
            # Second, local search: try to swap edges to enable further reduction
            # We iterate until no improvement is found
            improved = True
            while improved:
                improved = False
                if time.time() > deadline - 2.0:
                    break
                
                for i in range(len(current_selected)):
                    if time.time() > deadline - 2.0:
                        break
                    
                    # Calculate coverage if we remove current_selected[i]
                    union_without_i = 0
                    for j in range(len(current_selected)):
                        if j != i:
                            union_without_i |= edge_cover_masks[current_selected[j]]
                    
                    # If it still covers, it's redundant (should have been caught above, but maybe not if we swapped)
                    if union_without_i == full_mask:
                        current_selected.pop(i)
                        improved = True
                        break
                    
                    # Find missing colorings
                    missing = full_mask ^ union_without_i
                    
                    # Try to find an edge that covers all missing colorings
                    # and is not already in the set
                    for e_idx in range(num_edges):
                        if e_idx in current_selected:
                            continue
                        
                        if (edge_cover_masks[e_idx] & missing) == missing:
                            # Swap edge i with e_idx
                            current_selected[i] = e_idx
                            improved = True
                            break
                    
                    if improved:
                        break
            
            # 5. Evaluate result
            if len(current_selected) < min_edges_found:
                min_edges_found = len(current_selected)
                best_edges = [list(all_edges[i]) for i in current_selected]
                best_n = n

        except Exception:
            continue
    
    # Fallback if nothing found (should not happen)
    if best_edges is None:
        # Return a known valid instance, e.g. complete graph on 9 vertices?
        # K9^5 has 126 edges. It is definitely non-2-colorable.
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n

    return {'n': best_n, 'edges': best_edges}
