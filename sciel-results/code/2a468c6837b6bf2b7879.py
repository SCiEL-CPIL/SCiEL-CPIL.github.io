"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Precompute popcount for all masks
    # Using a list of ints for speed
    total_masks = 1 << n
    popcount = [0] * total_masks
    for i in range(1, total_masks):
        popcount[i] = popcount[i >> 1] + (i & 1)
        
    best_subset = list(range(min_v))
    best_distinct = 0
    
    # We want to maximize distinct degrees.
    # Max possible is n (17).
    # We iterate all masks.
    # Optimization: If we find a mask with distinct == size, and size is large, it's good.
    # But we just need the global max.
    
    # Iterate masks from largest to smallest?
    # Actually, order doesn't matter for correctness, but might for early exit if we hit 17.
    # Let's just iterate all.
    
    # To speed up, we can skip masks with size < min_v
    # We can also break early if best_distinct == n (17), as that's the max possible.
    
    for mask in range(total_masks - 1, 0, -1):
        size = popcount[mask]
        if size < min_v:
            continue
            
        # Upper bound on distinct degrees for this mask is 'size'
        # If size <= best_distinct, we can't beat the current best with this mask
        if size <= best_distinct:
            continue
            
        # Calculate degrees for vertices in mask
        # Use a set to track distinct degrees
        # Since size <= 17, a set is fine.
        # Or a small list and sort/unique, but set is O(1) avg.
        deg_set = set()
        
        # Iterate vertices
        # Using bit manipulation to iterate set bits is faster than range(n) with check
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m ^= lsb
            
            # Degree of v in mask is popcount(adj_mask[v] & mask)
            deg = popcount[adj_mask[v] & mask]
            deg_set.add(deg)
            
            # Early break if we already have more distinct degrees than 'size' is impossible, 
            # but we can break if len(deg_set) + (remaining vertices) <= best_distinct?
            # Not easily doable without knowing remaining.
            # However, if len(deg_set) == size, we can't add more distinct degrees.
            # But we still need to finish to know the size? No, size is known.
            # If len(deg_set) == size, distinct is size.
            
        distinct = len(deg_set)
        
        if distinct > best_distinct:
            best_distinct = distinct
            # Convert mask to list of vertices
            subset = []
            m = mask
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                subset.append(v)
                m ^= lsb
            best_subset = subset
            
            # If we reached the theoretical maximum for this size, and size is max possible (17), 
            # or if distinct == n, we can return immediately.
            if best_distinct == n:
                return best_subset
                
    return best_subset

# EVOLVE-BLOCK-END
