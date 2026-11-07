"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as bitmasks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    best_subset = list(range(8))
    best_count = 0
    
    deadline = time.time() + 82  # Safety margin before 90s kill
    
    # Precompute vertex masks
    vmask = [1 << i for i in range(n)]
    
    # Enumerate all subsets of size 8 through 17
    # Larger subsets first: more vertices means more potential distinct degrees
    for size in range(n, 7, -1):
        if time.time() > deadline:
            break
        # Early exit: max possible distinct degrees for size k is k
        if size <= best_count:
            # Can't do better than what we have with fewer vertices
            # But we still need at least 8, so skip sizes <= best_count
            # Actually we can skip this size entirely since max distinct = size <= best_count
            continue
        for subset in combinations(range(n), size):
            if time.time() > deadline:
                break
            smask = 0
            for v in subset:
                smask |= vmask[v]
            deg_set = set()
            for v in subset:
                d = (adj_mask[v] & smask).bit_count()
                deg_set.add(d)
            cnt = len(deg_set)
            if cnt > best_count:
                best_count = cnt
                best_subset = list(subset)
                # If we achieved the theoretical max for this size, 
                # no need to check other subsets of same size
                if cnt == size:
                    break
    
    return [int(v) for v in best_subset]
# EVOLVE-BLOCK-END
