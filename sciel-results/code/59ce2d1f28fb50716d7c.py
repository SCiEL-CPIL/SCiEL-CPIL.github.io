"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_min_vertices = P['minimum_vertices']

# Precompute adjacency as bitmask integers
_adj_mask = [0] * _N
for (u, v) in P['edges']:
    _adj_mask[u] |= (1 << v)
    _adj_mask[v] |= (1 << u)

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin under 90s kill
    best_score = -1
    best_mask = 0
    
    n = _N
    min_v = _min_vertices
    
    # Enumerate all subsets
    # We can iterate from (1 << min_v) to (1 << n) - 1? No, that doesn't guarantee popcount >= min_v.
    # We must check popcount.
    # Total subsets: 2^17 = 131072. This is very fast.
    
    for mask in range(1, 1 << n):
        if time.time() > deadline:
            break
            
        # Quick check on popcount
        # Using int.bit_count() which is O(1) in CPython 3.10+
        # Fallback for older versions if needed, but assuming modern Python.
        pc = mask.bit_count()
        if pc < min_v:
            continue
            
        # Calculate distinct degrees
        degs = set()
        # Iterate through set bits
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            v = lsb.bit_length() - 1
            # Degree of v in induced subgraph is popcount(adj[v] & mask)
            deg = (_adj_mask[v] & mask).bit_count()
            degs.add(deg)
            m ^= lsb
            
        score = len(degs)
        if score > best_score:
            best_score = score
            best_mask = mask
            
            # Theoretical maximum distinct degrees for a graph of size k is k.
            # If we reach k distinct degrees, we can't do better for this k.
            # But larger k can have larger distinct degrees.
            # Max possible score overall is n (17).
            if best_score == n:
                break
                
    # Convert best_mask to list of vertex indices
    result = []
    for v in range(n):
        if best_mask & (1 << v):
            result.append(v)
            
    # If no valid subset found (shouldn't happen), return first min_v vertices
    if not result:
        result = list(range(min_v))
        
    return result
# EVOLVE-BLOCK-END
