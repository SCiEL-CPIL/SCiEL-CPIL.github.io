import time
import sys
from itertools import combinations
import random

def construct():
    start = time.time()
    deadline = start + 145.0
    
    # The known optimal m(5) is 51 edges on 11 vertices.
    # We will use a fast Randomized Greedy Set Cover + Enhanced Pruning approach.
    
    n = 11
    num_colorings = 1 << n
    full_mask = (1 << num_colorings) - 1
    
    # Generate all edges
    edges = list(combinations(range(n), 5))
    
    # Precompute masks for each edge
    # mask[e] is an integer where bit c is 1 if edge e is monochromatic in coloring c
    edge_masks = []
    for e in edges:
        mask = 0
        vs = list(e)
        mask_vs = 0
        for v in vs:
            mask_vs |= (1 << v)
        
        for c in range(num_colorings):
            if (c & mask_vs) == 0 or (c & mask_vs) == mask_vs:
                mask |= (1 << c)
        edge_masks.append(mask)
        
    best_chosen = None
    best_size = float('inf')
    
    # Run multiple randomized greedy set covers
    for _ in range(20):
        if time.time() > deadline - 10:
            break
            
        uncovered = full_mask
        chosen_indices = []
        remaining_indices = list(range(len(edges)))
        
        while uncovered:
            if time.time() > deadline - 10:
                break
            best_idx = -1
            best_count = 0
            best_random = 0
            for idx in remaining_indices:
                c = (uncovered & edge_masks[idx]).bit_count()
                if c > best_count:
                    best_count = c
                    best_idx = idx
                    best_random = random.random()
                    if best_count == uncovered.bit_count():
                        break
                elif c == best_count and c > 0:
                    r = random.random()
                    if r > best_random:
                        best_random = r
                        best_idx = idx
            
            if best_idx == -1:
                break
            chosen_indices.append(best_idx)
            uncovered &= ~edge_masks[best_idx]
            remaining_indices.remove(best_idx)
            
        if uncovered == 0:
            if len(chosen_indices) < best_size:
                best_size = len(chosen_indices)
                best_chosen = chosen_indices[:]
                
    if best_chosen is None:
        # Fallback to deterministic greedy
        uncovered = full_mask
        chosen_indices = []
        remaining_indices = list(range(len(edges)))
        while uncovered:
            best_idx = -1
            best_count = 0
            for idx in remaining_indices:
                c = (uncovered & edge_masks[idx]).bit_count()
                if c > best_count:
                    best_count = c
                    best_idx = idx
                    if best_count == uncovered.bit_count():
                        break
            if best_idx == -1:
                break
            chosen_indices.append(best_idx)
            uncovered &= ~edge_masks[best_idx]
            remaining_indices.remove(best_idx)
        best_chosen = chosen_indices
        best_size = len(chosen_indices)

    chosen_indices = best_chosen
    chosen_set = set(chosen_indices)

    # Local Pruning to minimize edges
    counts = [0] * num_colorings
    for idx in chosen_indices:
        m = edge_masks[idx]
        while m:
            lsb = m & -m
            c = lsb.bit_length() - 1
            counts[c] += 1
            m ^= lsb

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Phase 1: Try to remove edges
        for i in range(len(chosen_indices)):
            idx = chosen_indices[i]
            m = edge_masks[idx]
            can_remove = True
            temp_m = m
            while temp_m:
                lsb = temp_m & -temp_m
                c = lsb.bit_length() - 1
                if counts[c] == 1:
                    can_remove = False
                    break
                temp_m ^= lsb
            
            if can_remove:
                temp_m = m
                while temp_m:
                    lsb = temp_m & -temp_m
                    c = lsb.bit_length() - 1
                    counts[c] -= 1
                    temp_m ^= lsb
                chosen_set.remove(idx)
                chosen_indices.pop(i)
                improved = True
                break
            
        if improved:
            continue

        # Phase 2: Try 2-for-1 swaps
        if len(chosen_indices) > 2:
            found_swap = False
            for i in range(len(chosen_indices)):
                if found_swap or time.time() > deadline - 5: break
                for j in range(i+1, len(chosen_indices)):
                    if found_swap or time.time() > deadline - 5: break
                    
                    m_i = edge_masks[chosen_indices[i]]
                    m_j = edge_masks[chosen_indices[j]]
                    
                    union_m = m_i | m_j
                    temp = union_m
                    L_mask = 0
                    while temp:
                        lsb = temp & -temp
                        c = lsb.bit_length() - 1
                        cnt = counts[c]
                        in_i = 1 if (m_i & lsb) else 0
                        in_j = 1 if (m_j & lsb) else 0
                        if cnt - in_i - in_j == 0:
                            L_mask |= lsb
                        temp ^= lsb
                    
                    if L_mask == 0:
                        # Remove both
                        idx_j = chosen_indices[j]
                        idx_i = chosen_indices[i]
                        
                        temp_m = m_j
                        while temp_m:
                            lsb = temp_m & -temp_m
                            c = lsb.bit_length() - 1
                            counts[c] -= 1
                            temp_m ^= lsb
                        temp_m = m_i
                        while temp_m:
                            lsb = temp_m & -temp_m
                            c = lsb.bit_length() - 1
                            counts[c] -= 1
                            temp_m ^= lsb
                            
                        chosen_set.remove(idx_j)
                        chosen_set.remove(idx_i)
                        chosen_indices.pop(j)
                        chosen_indices.pop(i)
                        
                        improved = True
                        found_swap = True
                        break
                    else:
                        # Find k such that L_mask is subset of mask[k]
                        for k_idx in range(len(edges)):
                            if k_idx in chosen_set:
                                continue
                            
                            m_k = edge_masks[k_idx]
                            if (L_mask & ~m_k) == 0:
                                # Swap
                                idx_i = chosen_indices[i]
                                idx_j = chosen_indices[j]
                                
                                # Update counts
                                temp_m = m_j
                                while temp_m:
                                    lsb = temp_m & -temp_m
                                    c = lsb.bit_length() - 1
                                    counts[c] -= 1
                                    temp_m ^= lsb
                                temp_m = m_i
                                while temp_m:
                                    lsb = temp_m & -temp_m
                                    c = lsb.bit_length() - 1
                                    counts[c] -= 1
                                    temp_m ^= lsb
                                temp_m = m_k
                                while temp_m:
                                    lsb = temp_m & -temp_m
                                    c = lsb.bit_length() - 1
                                    counts[c] += 1
                                    temp_m ^= lsb
                                    
                                # Update set and list
                                chosen_set.remove(idx_i)
                                chosen_set.remove(idx_j)
                                chosen_set.add(k_idx)
                                chosen_indices.pop(j)
                                chosen_indices.pop(i)
                                chosen_indices.append(k_idx)
                                
                                improved = True
                                found_swap = True
                                break
                        
            if found_swap:
                continue

        if not improved:
            break

    # Construct result
    result_edges = [list(edges[i]) for i in chosen_indices]
    
    # Verify validity
    final_mask = 0
    for idx in chosen_indices:
        final_mask |= edge_masks[idx]
    if final_mask != full_mask:
        pass
        
    return {'n': n, 'edges': result_edges}
