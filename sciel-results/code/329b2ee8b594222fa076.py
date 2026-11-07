"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 90s. We finish in <1s.
    deadline = start_time + 80.0
    
    n = P['n']
    edges = P['edges']
    min_verts = P['minimum_vertices']
    
    # Build adjacency bitmasks for fast degree calculation
    # adj_mask[v] is an integer where bit u is 1 if (u,v) is an edge
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    best_subset = None
    best_count = 0
    
    # Iterate through all possible subset sizes from largest to smallest
    # Largest subsets have more vertices, potentially more distinct degrees
    for k in range(n, min_verts - 1, -1):
        if time.time() > deadline:
            break
            
        for subset in combinations(range(n), k):
            if time.time() > deadline:
                break
                
            # Create a bitmask for the current subset
            subset_mask = 0
            for v in subset:
                subset_mask |= (1 << v)
                
            degrees = []
            for v in subset:
                # Count neighbors of v within the subset
                # adj_mask[v] & subset_mask gives the neighbors of v that are in the subset
                deg = bin(adj_mask[v] & subset_mask).count('1')
                degrees.append(deg)
                
            distinct_degrees = len(set(degrees))
            
            if distinct_degrees > best_count:
                best_count = distinct_degrees
                best_subset = list(subset)
                
                # Optional: Early exit if we reach the theoretical maximum for this k
                # Max distinct degrees in k vertices is k.
                # If we find k distinct degrees, we can't do better for this k.
                # But we still need to check smaller k? No, smaller k has lower max possible distinct degrees.
                # Wait, max distinct degrees for size k is k.
                # If we find k distinct degrees in a subset of size k, that's the max for that size.
                # However, a smaller subset might have more distinct degrees?
                # No, max distinct degrees for size m is m.
                # If k=17 and we find 17 distinct degrees, that's the global max possible for any subset.
                # If k=8 and we find 8 distinct degrees, that's max for size 8.
                # But 8 < 17. So a subset of size 17 with 17 distinct degrees is better than size 8 with 8.
                # So if we find best_count == k, and we are iterating k downwards, 
                # we can't find a higher count in smaller k.
                # So we can break out of the inner loop, and potentially the outer loop if best_count == n?
                # Actually, if best_count == k, and we go to k-1, max possible is k-1 < k.
                # So we can stop entirely if best_count == k.
                if best_count == k:
                    return best_subset

    if best_subset is None:
        # Fallback: return first min_verts vertices
        best_subset = list(range(min_verts))
        
    # Ensure types are Python int and list
    return [int(v) for v in best_subset]
# EVOLVE-BLOCK-END
