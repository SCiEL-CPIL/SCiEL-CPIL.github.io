"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Build adjacency bitmask for fast degree computation
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def count_distinct_degrees_mask(mask):
        """Count distinct degrees in the induced subgraph using bitmask."""
        degrees = set()
        for v in range(n):
            if mask & (1 << v):
                deg = (adj_mask[v] & mask).bit_count()
                degrees.add(deg)
        return len(degrees)
    
    best_count = 0
    best_mask = 0
    
    # Exhaustive search over all subsets of size >= min_v
    # Total subsets for sizes 8..17 = sum(C(17,k), k=8..17) = 65536, very manageable
    for size in range(min_v, n + 1):
        if time.time() > deadline:
            break
        for subset in itertools.combinations(range(n), size):
            if time.time() > deadline:
                break
            mask = 0
            for v in subset:
                mask |= (1 << v)
            distinct = count_distinct_degrees_mask(mask)
            if distinct > best_count:
                best_count = distinct
                best_mask = mask
    
    # Convert best_mask to list of vertex indices
    result = []
    for v in range(n):
        if best_mask & (1 << v):
            result.append(int(v))
    
    return result
# EVOLVE-BLOCK-END
