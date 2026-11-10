"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute validity of all masks
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    popcount = [0] * (1 << n)
    
    # Precompute popcount
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
        
    is_clique[0] = True
    is_ind[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest]:
            if (adj[v] & rest) == rest:
                is_clique[mask] = True
                
        if is_ind[rest]:
            if (adj[v] & rest) == 0:
                is_ind[mask] = True

    # Collect valid masks
    valid_masks = []
    for mask in range(1, 1 << n):
        if is_clique[mask] or is_ind[mask]:
            valid_masks.append(mask)
            
    valid_masks.sort(key=lambda x: popcount[x], reverse=True)
    
    masks_by_vertex = [[] for _ in range(n)]
    for mask in valid_masks:
        for v in range(n):
            if mask & (1 << v):
                masks_by_vertex[v].append(mask)
                
    for v in range(n):
        masks_by_vertex[v].sort(key=lambda x: popcount[x], reverse=True)
        
    best_count = [n + 1]
    best_solution = [None]
    
    # Precompute max valid submask size for each mask
    max_valid_size = [0] * (1 << n)
    for mask in range(1, 1 << n):
        if is_clique[mask] or is_ind[mask]:
            max_valid_size[mask] = popcount[mask]
        else:
            max_sz = 0
            m = mask
            while m:
                lsb = m & -m
                sub = mask ^ lsb
                if max_valid_size[sub] > max_sz:
                    max_sz = max_valid_size[sub]
                m ^= lsb
            max_valid_size[mask] = max_sz
            
    def backtrack(remaining_mask, num_parts, current_parts):
        if remaining_mask == 0:
            if num_parts < best_count[0]:
                best_count[0] = num_parts
                best_solution[0] = current_parts.copy()
            return
            
        max_sz = max_valid_size[remaining_mask]
        if max_sz == 0:
            return
            
        lower_bound = num_parts + math.ceil(popcount[remaining_mask] / max_sz)
        if lower_bound >= best_count[0]:
            return
            
        best_v = -1
        min_count = float('inf')
        
        m = remaining_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            count = 0
            for mask in masks_by_vertex[v]:
                if (mask & remaining_mask) == mask:
                    count += 1
            if count < min_count:
                min_count = count
                best_v = v
            m ^= lsb
            
        for mask in masks_by_vertex[best_v]:
            if (mask & remaining_mask) == mask:
                new_remaining = remaining_mask ^ mask
                current_parts.append(mask)
                backtrack(new_remaining, num_parts + 1, current_parts)
                current_parts.pop()
                
    backtrack(full_mask, 0, [])
    
    if best_solution[0] is not None:
        result = []
        for mask in best_solution[0]:
            part = []
            for v in range(n):
                if mask & (1 << v):
                    part.append(v)
            result.append(part)
        return result
    else:
        return [[v] for v in range(n)]

# EVOLVE-BLOCK-END
