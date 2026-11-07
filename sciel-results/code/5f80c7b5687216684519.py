"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    min_v = P['minimum_vertices']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute popcount for all possible masks? No, just use bin(x).count('1') or x.bit_count()
    # Python 3.10+ has int.bit_count(). Let's assume standard library is recent enough or use fallback.
    # To be safe, we can use a lookup table or bin().count. bin().count is fast enough for 17 bits.
    
    best_mask = 0
    max_distinct = -1
    
    # Iterate through all subsets
    # Optimization: We can prune if we already found 17 distinct degrees (impossible for 17 vertices? Max degree 16, so 17 distinct values 0..16 is possible only if all degrees are unique. In a 17-vertex graph, degrees are 0..16. If all are present, count is 17. But induced subgraph size k has degrees 0..k-1. Max distinct is k. So max possible is 17.)
    
    for mask in range(1 << n):
        # Check minimum vertices constraint
        # bit_count is fast
        k = mask.bit_count()
        if k < min_v:
            continue
            
        # Calculate degrees for vertices in mask
        # We only care about distinct degrees
        degrees = set()
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            u = lsb.bit_length() - 1
            # Compute degree of u in induced subgraph
            # Neighbors of u in mask
            deg_mask = adj[u] & mask
            deg = deg_mask.bit_count()
            degrees.add(deg)
            # Early exit if we found all possible distinct degrees for this k?
            # Max possible distinct degrees is k (0 to k-1).
            # If len(degrees) == k, we can't do better for this k, but maybe a larger k yields more?
            # Actually, we just want global max.
            # We can break if len(degrees) == k, as this is the max for this subset size.
            # But we still need to check other masks.
            m ^= lsb
            
        if len(degrees) > max_distinct:
            max_distinct = len(degrees)
            best_mask = mask
            # If we reached the theoretical maximum for any subset size, we can stop?
            # Theoretical max for subset of size k is k.
            # Global max is 17 (if a 17-vertex induced subgraph has all distinct degrees).
            # But we must have at least 8 vertices.
            # If max_distinct == 17, we can break.
            if max_distinct == n:
                break

    # Convert best_mask to list of vertices
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        u = lsb.bit_length() - 1
        result.append(u)
        m ^= lsb
        
    # Ensure types are correct: list of Python ints
    return result

# EVOLVE-BLOCK-END
