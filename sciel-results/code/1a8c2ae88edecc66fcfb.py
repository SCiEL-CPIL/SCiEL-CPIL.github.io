"""Exhaustive search for maximum distinct degrees in induced subgraph with pruning."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def construct():
    deadline = time.time() + 80.0  # safety margin under 90s wall-clock
    n = P['n']
    edges = P['edges']
    minv = P['minimum_vertices']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
        
    vertices = list(range(n))
    best_score = -1
    best_vertices = list(range(minv)) # fallback
    
    # Precompute bit_count for speed
    def _bit_count(x):
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')
    
    # Iterate subset sizes from largest to smallest
    # Max distinct degrees for size k is k.
    # We want to maximize this count.
    for size in range(n, minv - 1, -1):
        if time.time() > deadline:
            break
            
        # If we already have a score >= size, no subset of this size can beat it
        # (since max distinct degrees is size). 
        # However, we are iterating down, so if best_score is 10, and we are at size 9, 
        # max possible is 9, which is < 10, so we can stop.
        if best_score >= size:
            break
            
        for combo in itertools.combinations(vertices, size):
            if time.time() > deadline:
                return [int(v) for v in best_vertices]
                
            # Build mask for this combo
            mask = 0
            for v in combo:
                mask |= (1 << v)
                
            # Quick upper bound: count how many vertices have unique-ish degrees
            # If many vertices share the same degree, distinct count is limited
            # Compute degrees and check if we can possibly beat best_score
            degs = set()
            for v in combo:
                x = adj[v] & mask
                c = _bit_count(x)
                degs.add(c)
                # Early exit: if we've already seen all possible distinct values
                # and can't beat best_score
                if len(degs) + (size - len(degs) - len(degs)) <= best_score and len(degs) < best_score:
                    # This is a loose bound; let's just continue
                    pass
                
            distinct_count = len(degs)
            
            if distinct_count > best_score:
                best_score = distinct_count
                best_vertices = list(combo)
                
    # Ensure exact Python int types
    out = [int(v) for v in best_vertices]
    return out
# EVOLVE-BLOCK-END
