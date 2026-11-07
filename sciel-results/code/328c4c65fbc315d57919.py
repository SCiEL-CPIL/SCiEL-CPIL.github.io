import itertools
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        
    best_score = 0
    best_subset = list(range(8)) # Fallback
    
    vertices = list(range(n))
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0 # 50s margin out of 90s
    
    # We iterate k from 17 down to 8.
    # If we find 9 distinct degrees, we stop.
    # 9 is the max possible (degrees 0..8).
    
    for k in range(n, 7, -1):
        if time.time() > deadline:
            break
            
        for subset in itertools.combinations(vertices, k):
            if time.time() > deadline:
                break
                
            mask = 0
            for v in subset:
                mask |= 1 << v
                
            degrees = set()
            for v in subset:
                # Use bit_count for speed
                deg = (adj[v] & mask).bit_count()
                degrees.add(deg)
                
            distinct_count = len(degrees)
            if distinct_count > best_score:
                best_score = distinct_count
                best_subset = list(subset)
                
                if best_score == 9:
                    # Max possible distinct degrees found
                    return best_subset
                    
    return best_subset
