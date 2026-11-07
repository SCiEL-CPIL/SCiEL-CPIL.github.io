"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    best_count = 0
    best_mask = 0
    
    # Iterate through all possible subsets
    # We only care about subsets with at least 8 vertices
    # Total subsets 2^17 = 131072, which is very small
    for mask in range(1, 1 << n):
        # Quick check on popcount
        # Using bit_count() for speed
        if mask.bit_count() < 8:
            continue
            
        # Calculate degrees for vertices in the mask
        deg_set = set()
        # We can iterate through set bits or just all vertices
        # Iterating all n vertices is fine (17 checks)
        for i in range(n):
            if (mask >> i) & 1:
                # Degree of i in induced subgraph is number of neighbors in mask
                deg = (adj[i] & mask).bit_count()
                deg_set.add(deg)
                
        count = len(deg_set)
        if count > best_count:
            best_count = count
            best_mask = mask
            
            # The maximum possible distinct degrees is the size of the subset
            # If we reach the size of the subset, we can't do better for this size
            # But we might find a larger subset with even more distinct degrees?
            # No, max distinct degrees for a subset of size k is k.
            # So if we find a subset of size k with k distinct degrees, 
            # we can only beat it with a subset of size > k with > k distinct degrees.
            # Since max size is 17, if we find 17 distinct degrees, we can stop.
            if best_count == n:
                break
                
    # Convert best_mask to list of vertices
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i)
            
    return result
# EVOLVE-BLOCK-END
