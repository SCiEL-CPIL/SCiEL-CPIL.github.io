"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit
    
    n = P['n']
    min_k = P['minimum_vertices']
    
    # Build adjacency as bitmask for fast degree computation
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute: for each vertex, its adjacency bitmask
    # For a subset represented as bitmask, degree of v in subset = popcount(adj_mask[v] & subset_mask)
    
    # Use Python's int.bit_count() (Python 3.10+) for fast popcount
    # Fallback: bin(x).count('1')
    
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')
    
    best_count = 0
    best_subset = None
    
    # Enumerate all subsets of size >= min_k
    # Total subsets of size >= 8 from 17: sum(C(17,k) for k=8..17) = 131072 - sum(C(17,k) for k=0..7)
    # C(17,0)=1, C(17,1)=17, C(17,2)=136, C(17,3)=680, C(17,4)=2380, C(17,5)=6188, C(17,6)=12376, C(17,7)=19448
    # Sum = 41226, so subsets of size >= 8 = 131072 - 41226 = 89846
    # This is very manageable
    
    for k in range(n, min_k - 1, -1):
        if time.time() > deadline:
            break
        for combo in combinations(range(n), k):
            if time.time() > deadline:
                break
            # Compute distinct degrees in induced subgraph
            subset_mask = 0
            for v in combo:
                subset_mask |= (1 << v)
            
            degs = set()
            for v in combo:
                d = popcount(adj_mask[v] & subset_mask)
                degs.add(d)
            
            cnt = len(degs)
            if cnt > best_count:
                best_count = cnt
                best_subset = combo
                # Can't do better than k distinct degrees (one per vertex)
                if best_count >= k:
                    break
        if best_count >= k:
            break
    
    if best_subset is None:
        return list(range(min_k))
    
    return list(best_subset)
# EVOLVE-BLOCK-END
