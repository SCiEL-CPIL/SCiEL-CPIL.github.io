"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_verts = P['minimum_vertices']
    
    # Build adjacency bitmask for each vertex
    # adj_mask[v] is an integer where bit u is set if (v, u) is an edge
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    best_subset = list(range(min_verts))
    best_distinct_count = 0
    
    # Time limit guard
    start_time = time.time()
    timeout = 80.0  # 90s limit, keep 10s margin
    
    # Search subset sizes from largest to smallest
    # For size k, max possible distinct degrees is k
    for k in range(n, min_verts - 1, -1):
        if time.time() - start_time > timeout:
            break
            
        # If we already found k distinct degrees, no subset of size <= k can beat it
        # (Max distinct degrees for size j is j. If best >= k, and we are checking j <= k, 
        #  then max possible for j is j <= k <= best. So we cannot improve.)
        if best_distinct_count >= k:
            break
            
        # Iterate through all combinations of size k
        for subset in itertools.combinations(range(n), k):
            if time.time() - start_time > timeout:
                break
                
            # Build bitmask for the subset
            mask = 0
            for v in subset:
                mask |= (1 << v)
                
            # Calculate degrees in the induced subgraph
            degrees = []
            for v in subset:
                # Count neighbors of v that are also in the subset
                deg = (adj_mask[v] & mask).bit_count()
                degrees.append(deg)
                
            distinct_degrees = len(set(degrees))
            
            if distinct_degrees > best_distinct_count:
                best_distinct_count = distinct_degrees
                best_subset = list(subset)
                
                # If all degrees are distinct, this is the maximum possible for this k
                if distinct_degrees == k:
                    # We can't do better for this k. 
                    # But can we do better for larger k? No, we are going down.
                    # Can we do better for smaller k? Max for smaller k is smaller.
                    # So if we find k distinct degrees in a size k subset, 
                    # and we started from n down to k, this is the global max 
                    # IF k is the largest size we've checked so far? 
                    # No. We are checking k=17, then 16...
                    # If we find 17 distinct in k=17, global max is 17.
                    # If we find 16 distinct in k=17, we continue k=17.
                    # If we exhaust k=17 with max 16, we go to k=16.
                    # If we find 16 distinct in k=16, we can't beat 16.
                    # So if distinct_degrees == k, we can break the inner loop 
                    # because we can't get more than k for this size.
                    break
        
        # If we found k distinct degrees, we can break the outer loop 
        # because for any j < k, max distinct degrees is j < k.
        if best_distinct_count >= k:
            break
            
    return best_subset
# EVOLVE-BLOCK-END
