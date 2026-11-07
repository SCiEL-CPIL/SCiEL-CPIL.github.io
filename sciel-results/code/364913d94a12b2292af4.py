"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    best_count = -1
    best_subset = None
    
    # Deadline: 90s kill, use 80s safety margin
    deadline = time.time() + 80.0
    
    # Iterate over all masks with at least min_vertices bits set
    # Total masks 2^17 = 131,072. This is small enough to iterate fully if fast enough.
    # But we need to be safe.
    
    # Optimization: Precompute bit positions? No, bit_count is fast.
    
    for mask in range(1 << n):
        # Time check every so often to avoid overhead
        # Actually, checking time.time() is cheap enough in CPython for 131k iterations?
        # 131k * ~100ns = 13ms. So we can check every iteration or every 100.
        # Let's check every 1000 iterations to be safe, but 131k is small.
        # Let's just check every iteration for simplicity and safety.
        
        if time.time() > deadline:
            if best_subset is not None:
                subset = [i for i in range(n) if (mask >> i) & 1]
                return subset
            else:
                return list(range(min_vertices))
                
        k = mask.bit_count()
        if k < min_vertices:
            continue
            
        # Calculate distinct degrees
        # We need to count distinct values of (adj_mask[v] & mask).bit_count() for v in mask.
        
        # Optimization: 
        # If k == 17, max distinct is 17.
        # If we find 17, we can stop? No, we need to return the subset.
        # But 17 is the theoretical max. If we find it, we can return immediately?
        # Yes, because we can't do better than k distinct degrees for a subset of size k.
        # And max k is 17. So 17 is the global max.
        
        degrees_seen = set()
        # Iterate bits in mask
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            
            deg = (adj_mask[v] & mask).bit_count()
            degrees_seen.add(deg)
            
            # Early termination if we already have k distinct degrees
            if len(degrees_seen) == k:
                break
                
        distinct_count = len(degrees_seen)
        
        if distinct_count > best_count:
            best_count = distinct_count
            best_subset = mask
            
            # If we reached the theoretical maximum for this k, and k is the max possible (17),
            # we can stop?
            # Actually, if distinct_count == k, we can't improve for this k.
            # But we might find a larger k with more distinct degrees?
            # Max distinct for k is k. So if we have k=10 with 10 distinct, and k=11 with 11 distinct,
            # 11 is better.
            # So we should continue until we hit k=17 and find 17 distinct?
            # Or if we find 17 distinct at k=17, we can stop.
            
            if distinct_count == 17:
                # Global max found
                subset = [i for i in range(n) if (best_subset >> i) & 1]
                return subset
    
    if best_subset is not None:
        subset = [i for i in range(n) if (best_subset >> i) & 1]
        return subset
    else:
        return list(range(min_vertices))
# EVOLVE-BLOCK-END
