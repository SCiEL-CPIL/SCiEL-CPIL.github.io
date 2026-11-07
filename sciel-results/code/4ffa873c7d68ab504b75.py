import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Precompute adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    best_score = 0
    best_combo = ()
    start_time = time.time()
    deadline = start_time + 45.0
    
    # Iterate from largest to smallest size; prune sizes that cannot beat best_score
    for size in range(n, min_v - 1, -1):
        # A subset of size 'size' can have at most 'size' distinct degrees.
        # If we already have best_score >= size, skip this size entirely.
        if best_score >= size:
            continue
            
        for combo in combinations(range(n), size):
            # Check time
            if time.time() > deadline:
                if best_combo:
                    return list(best_combo)
                return list(range(min_v))
                
            # Build mask for fast degree calculation
            mask = 0
            for v in combo:
                mask |= (1 << v)
                
            # Calculate distinct degrees
            degrees = set()
            for v in combo:
                deg = (adj[v] & mask).bit_count()
                degrees.add(deg)
                
            score = len(degrees)
            if score > best_score:
                best_score = score
                best_combo = combo
    
    return list(best_combo)
