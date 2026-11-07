"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.monotonic() + 80.0

    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    min_size = P['minimum_vertices']

    best_deg = 0
    best_vertices = list(range(min_size))

    # Precompute popcount for all masks up to 2^n to avoid repeated bit_count calls
    # Actually, Python's bit_count is fast enough, but let's try a different approach.
    
    # Strategy: enumerate all subsets of size >= min_size
    # Use itertools.combinations for potentially faster enumeration of fixed-size subsets
    # But the current bitmask approach is already quite fast.
    
    # Let's try a different optimization: precompute the degree contribution of each vertex
    # for each possible subset... that's too expensive (2^17 * 17).
    
    # Instead, let's use a smarter enumeration order or early termination.
    # The key insight: we want to maximize distinct degrees. In a subset of size k,
    # the maximum possible distinct degrees is k (if all degrees are different).
    # For k=8, max is 8. For larger k, max is k.
    
    # Let's try enumerating subsets of exactly min_size first, then larger sizes.
    # This might find good solutions faster.
    
    import itertools
    
    for size in range(min_size, n + 1):
        if time.monotonic() > deadline:
            break
        
        for combo in itertools.combinations(range(n), size):
            if time.monotonic() > deadline:
                break
            
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            # Compute distinct degrees
            degs = set()
            for v in combo:
                d = (adj[v] & mask).bit_count()
                degs.add(d)
            
            nd = len(degs)
            if nd > best_deg:
                best_deg = nd
                best_vertices = list(combo)
                # Early termination if we've achieved the maximum possible
                if best_deg == size:
                    break
        else:
            continue
        break
    
    return [int(v) for v in best_vertices]
# EVOLVE-BLOCK-END
