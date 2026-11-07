"""Improved program for the suite task; optimized exhaustive search with bitmasks."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Build adjacency as bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    best_subset = list(range(min_v))
    best_distinct = 0
    start_time = time.time()
    deadline = start_time + 50 # Generous safety margin
    
    # We iterate over all possible subset sizes from min_v to n
    # Using bitmask representation for the subset itself allows fast degree calculation
    # However, generating combinations of indices is still the easiest way to manage the "subset" concept
    # For each subset, we compute a mask.
    
    for k in range(min_v, n + 1):
        for subset in itertools.combinations(range(n), k):
            if time.time() > deadline:
                break
            
            # Create a bitmask for this subset
            mask = 0
            for v in subset:
                mask |= (1 << v)
                
            degrees = []
            for v in subset:
                # Count bits in (adj[v] & mask)
                # Python 3.10+ has int.bit_count()
                # Fallback: bin(x).count('1') is slower
                # Let's assume Python 3.10+ or use a helper
                # The environment usually supports modern Python features
                deg = (adj[v] & mask).bit_count()
                degrees.append(deg)
                
            distinct = len(set(degrees))
            
            if distinct > best_distinct:
                best_distinct = distinct
                best_subset = list(subset)
                
                # Theoretical max distinct degrees for k vertices is k
                # If we reach k, we can't do better for this k
                # But we might find k+1 for larger k? No, max is size of subset.
                # So if distinct == k, it's the best possible for this size.
                # However, a larger subset might have more distinct degrees (e.g. 9 distinct in 10 vertices vs 8 in 8 vertices).
                # So we continue searching all k.
                if best_distinct == k:
                    # This is the maximum possible for this k.
                    # But we should still check if a larger k can yield more.
                    # We can't break the outer loop yet.
                    pass
        if time.time() > deadline:
            break
            
    return [int(x) for x in best_subset]
# EVOLVE-BLOCK-END
