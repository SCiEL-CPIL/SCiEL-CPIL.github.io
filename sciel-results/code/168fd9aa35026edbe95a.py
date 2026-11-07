"""Rewritten program with faster bitmask-based search for distinct degrees."""
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
    
    # Build adjacency as bitmasks for fast operations
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute full adjacency for quick degree-in-subset computation
    # For a given subset bitmask, degree of vertex v = popcount(adj_mask[v] & subset_mask)
    
    def popcount(x):
        return bin(x).count('1')
    
    def count_distinct_degrees(subset_mask):
        """Count distinct degrees in the induced subgraph of subset_mask."""
        degrees = set()
        for v in range(n):
            if subset_mask & (1 << v):
                deg = popcount(adj_mask[v] & subset_mask)
                degrees.add(deg)
        return len(degrees)
    
    # Strategy: Try all subsets of sizes from min_v to n
    # For small sizes, enumerate all combinations
    # For larger sizes, use smarter heuristics
    
    best_count = 0
    best_subset = None
    
    # First, try all subsets of size min_v (8) - there are C(17,8) = 24310
    for subset in itertools.combinations(range(n), min_v):
        if time.time() > deadline:
            break
        mask = 0
        for v in subset:
            mask |= (1 << v)
        distinct = count_distinct_degrees(mask)
        if distinct > best_count:
            best_count = distinct
            best_subset = list(subset)
    
    # If we found 8 distinct degrees, that's max for size 8
    if best_count >= min_v:
        # Try to find larger subsets with even more distinct degrees
        pass
    
    # Try sizes 9, 10, ... up to n
    for size in range(min_v + 1, n + 1):
        if time.time() > deadline:
            break
        
        # For larger sizes, C(17,9)=24310, C(17,10)=19448, etc.
        # Total from size 9 to 17 is manageable if we're efficient
        for subset in itertools.combinations(range(n), size):
            if time.time() > deadline:
                break
            mask = 0
            for v in subset:
                mask |= (1 << v)
            distinct = count_distinct_degrees(mask)
            if distinct > best_count:
                best_count = distinct
                best_subset = list(subset)
    
    # If no best found (shouldn't happen), fallback
    if best_subset is None:
        best_subset = list(range(min_v))
    
    return [int(v) for v in best_subset]
# EVOLVE-BLOCK-END
