"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 87  # safety margin before 90s kill
    n = P['n']
    edges = P['edges']
    min_verts = P['minimum_vertices']

    # Build adjacency bitmasks for fast induced-degree computation
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    best_subset = None
    best_distinct = 0

    # Precompute popcount lookup for 17-bit numbers
    # Actually, Python's int.bit_count() is faster than bin().count('1')
    
    # Try all subset sizes from n down to min_verts (larger subsets first)
    # since they have more potential for distinct degrees
    for k in range(n, min_verts - 1, -1):
        # Theoretical max distinct degrees for subset size k is min(k, k-1) = k-1
        # (degrees range from 0 to k-1, so at most k distinct values)
        max_possible = k  # degrees can be 0,1,...,k-1, so at most k distinct
        
        for combo in combinations(range(n), k):
            if time.time() > deadline:
                return [int(v) for v in best_subset] if best_subset else [int(v) for v in range(min_verts)]
            
            # Build subset bitmask
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            # Compute degrees within induced subgraph using bitmask & bit_count
            degrees = set()
            for v in combo:
                deg = (adj_mask[v] & mask).bit_count()
                degrees.add(deg)
            
            distinct = len(degrees)
            if distinct > best_distinct:
                best_distinct = distinct
                best_subset = list(combo)
            
            # Early termination: if we achieved the theoretical max for this k
            if distinct >= max_possible:
                # Can't do better with this k, but larger k already checked
                # Since we're going from large to small, this is the best possible
                return [int(v) for v in best_subset]

    if best_subset is None:
        best_subset = list(range(min_verts))

    # Ensure types are correct
    return [int(v) for v in best_subset]

# EVOLVE-BLOCK-END
