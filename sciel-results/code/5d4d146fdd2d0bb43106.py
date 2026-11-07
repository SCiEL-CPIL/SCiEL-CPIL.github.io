import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_verts = P['minimum_vertices']
    
    # Build adjacency bitmasks
    adj_masks = [0] * n
    for u, v in edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    all_vertices = list(range(n))
    best_score = -1
    best_subset = None
    
    # Precompute popcount for all possible masks? No, 2^17 is 131072.
    # We can just use bit_count() which is fast in Python 3.8+ (actually 3.10+ for int.bit_count, but we can use bin(x).count('1') or a lookup table).
    # Python 3.10+ has int.bit_count(). Let's assume modern Python.
    # To be safe, I'll use a helper or just bin().count('1') which is fast enough for 17 bits.
    
    def popcount(x):
        return bin(x).count('1')
        
    deadline = time.time() + 50.0 # Safety margin
    
    # Iterate sizes from min_verts to n
    # To maximize distinct degrees, larger subsets have higher potential max (k-1).
    # But we must check all.
    
    # Optimization: If we find a subset with distinct_degrees == size - 1, 
    # that is the theoretical max for that size. 
    # However, we want global max across all sizes.
    # Max possible for n=17 is 16.
    # If we find 16, we can stop.
    
    for size in range(min_verts, n + 1):
        if time.time() > deadline:
            break
        for subset in combinations(all_vertices, size):
            if time.time() > deadline:
                break
            
            # Build mask for this subset
            mask = 0
            for v in subset:
                mask |= (1 << v)
                
            # Calculate degrees
            degs = set()
            for v in subset:
                # Count neighbors in mask
                # adj_masks[v] & mask gives neighbors of v that are in subset
                deg = popcount(adj_masks[v] & mask)
                degs.add(deg)
                
            score = len(degs)
            if score > best_score:
                best_score = score
                best_subset = subset
                
                # Early exit if we hit theoretical max for this size?
                # No, a larger size might have higher max.
                # But if best_score == size - 1, it's max for this size.
                # If we find 16 (max for 17), we can break all loops.
                if best_score == n - 1:
                    return list(best_subset)
                    
    if best_subset is None:
        return list(range(min_verts))
        
    return list(best_subset)
