"""Optimized program: bitmask-based exhaustive search with deadline guard."""
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_DEADLINE = 85  # seconds, safety margin from 90s kill

def _build_adj_bitmasks():
    n = P['n']
    adj_mask = [0] * n
    for a, b in P['edges']:
        adj_mask[a] |= (1 << b)
        adj_mask[b] |= (1 << a)
    return adj_mask

# Precompute popcount for 16-bit chunks (fallback if bit_count unavailable)
# But bit_count is available in Python 3.10+ and likely the environment.
# We'll use bit_count if available for speed.

def _popcount(x):
    return x.bit_count()

def _best_subset():
    n = P['n']
    min_v = P['minimum_vertices']
    adj_mask = _build_adj_bitmasks()
    start_time = time.time()
    
    best = None
    best_deg = -1
    
    # Iterate over all subsets using bitmask representation
    # Only consider subsets with at least min_v vertices
    for mask in range(1, 1 << n):
        # Check deadline periodically
        if (mask & 0xFFF) == 0 and time.time() - start_time > _DEADLINE:
            break
        
        # Count bits (number of vertices in subset)
        s = _popcount(mask)
        if s < min_v:
            continue
        
        # Compute degrees for each vertex in the subset
        degrees = set()
        m = mask
        while m:
            # Get lowest set bit
            lb = m & (-m)
            v = lb.bit_length() - 1
            # Degree of v in induced subgraph = popcount(adj_mask[v] & mask)
            d = _popcount(adj_mask[v] & mask)
            degrees.add(d)
            m ^= lb
        
        nd = len(degrees)
        if nd > best_deg:
            best_deg = nd
            best = [v for v in range(n) if (mask >> v) & 1]
        
        # Early termination: max possible distinct degrees in a subset of size s is s
        # If we've found s distinct degrees for the largest possible subset, we can't do better.
        if best_deg == n:
            break
    
    return best

# Compute once at module load
_BEST = _best_subset()

def construct():
    # Ensure we return a list of Python ints
    if _BEST is None:
        return list(range(P['minimum_vertices']))
    result = [int(v) for v in _BEST]
    # Safety: ensure at least minimum_vertices
    if len(result) < P['minimum_vertices']:
        result = list(range(P['minimum_vertices']))
    return result
