import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_n = None
    best_edges = None
    best_count = float('inf')
    
    # Try n from 9 to 16
    for n in range(9, 17):
        if time.time() > deadline:
            break
            
        # Generate all edges
        edges = list(combinations(range(n), 5))
        num_edges = len(edges)
        
        # Precompute vertex masks
        # Z_v: bitmask of colorings where vertex v is 0
        # O_v: bitmask of colorings where vertex v is 1
        # Total colorings: 2^n
        # We use Python integers as bitsets
        Z = [0] * n
        O = [0] * n
        
        # Optimization: build masks incrementally or using bit patterns
        # For v in range(n):
        #   pattern is 0 for first 2^(n-v-1) blocks of 2^(n-v-1) ... wait
        # Simpler:
        for v in range(n):
            mask_z = 0
            mask_o = 0
            for c in range(1 << n):
                if (c >> v) & 1 == 0:
                    mask_z |= (1 << c)
                else:
                    mask_o |= (1 << c)
            Z[v] = mask_z
            O[v] = mask_o
            
        # Compute edge masks
        edge_masks = []
        for e in edges:
            mask_z = (1 << (1 << n)) - 1  # All 1s
            mask_o = (1 << (1 << n)) - 1
            for v in e:
                mask_z &= Z[v]
                mask_o &= O[v]
            edge_masks.append(mask_z | mask_o)
            
        # Greedy Set Cover
        ALL_ONES = (1 << (1 << n)) - 1
        uncovered = ALL_ONES
        used_mask_indices = []
        remaining_indices = list(range(num_edges))
        
        while uncovered:
            if time.time() > deadline:
                break
                
            best_edge_idx = -1
            best_count_covers = 0
            
            for i in remaining_indices:
                covers = uncovered & edge_masks[i]
                count = covers.bit_count()
                if count > best_count_covers:
                    best_count_covers = count
                    best_edge_idx = i
                    
            if best_edge_idx == -1:
                break # Should not happen
                
            used_mask_indices.append(best_edge_idx)
            uncovered &= ~edge_masks[best_edge_idx]
            remaining_indices.remove(best_edge_idx)
            
        if not uncovered:
            # Try to reduce
            reduced = True
            while reduced:
                reduced = False
                for i in range(len(used_mask_indices)):
                    # Check if removing used_mask_indices[i] still covers all
                    or_mask = 0
                    for j in range(len(used_mask_indices)):
                        if i != j:
                            or_mask |= edge_masks[used_mask_indices[j]]
                    if or_mask == ALL_ONES:
                        # Remove it
                        used_mask_indices.pop(i)
                        reduced = True
                        break # restart loop
                        
            if len(used_mask_indices) < best_count:
                best_count = len(used_mask_indices)
                best_n = n
                best_edges = [list(edges[idx]) for idx in used_mask_indices]
                
    if best_edges is None:
        # Fallback to known valid construction or seed
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}
