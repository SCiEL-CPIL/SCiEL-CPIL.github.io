"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adj_bitmasks():
    """Build adjacency list as bitmasks for fast intersection operations."""
    n = P['n']
    adj_masks = [0] * n
    for u, v in P['edges']:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
    return adj_masks

def _popcount(x):
    """Fast population count for integers."""
    return bin(x).count('1')

def _distinct_degrees_for_mask(mask, adj_masks):
    """
    Calculate the number of distinct degrees in the induced subgraph defined by mask.
    Returns (distinct_count, degrees_set) for potential tie-breaking or analysis.
    """
    degrees = set()
    # Iterate over vertices in the mask
    m = mask
    while m:
        # Extract lowest set bit
        lsb = m & -m
        v = lsb.bit_length() - 1
        m ^= lsb
        
        # Degree of v in induced subgraph is popcount(adj[v] & mask)
        deg = _popcount(adj_masks[v] & mask)
        degrees.add(deg)
        
    return len(degrees)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = P['n']
    min_k = P['minimum_vertices']
    adj_masks = _build_adj_bitmasks()
    
    best_count = 0
    best_mask = 0
    
    # Iterate through all possible subsets (bitmasks)
    # We can optimize by skipping masks with popcount < min_k
    # Total masks: 2^17 = 131072. This is very fast.
    for mask in range(1, 1 << n):
        if time.time() > deadline:
            break
            
        # Quick check: if popcount < min_k, skip
        # bin(mask).count('1') is fast enough for 131k iterations
        if _popcount(mask) < min_k:
            continue
            
        cnt = _distinct_degrees_for_mask(mask, adj_masks)
        
        if cnt > best_count:
            best_count = cnt
            best_mask = mask
            
            # Theoretical max distinct degrees for a subset of size k is k.
            # If we reach this, we can't do better for this size, but maybe a larger size?
            # Actually, max distinct degrees is bounded by min(k, n-1).
            # If best_count == _popcount(mask), we have all distinct degrees.
            # It's unlikely to beat this with a different subset unless we find a larger k with k distinct degrees.
            # But since we are maximizing the COUNT, and count <= size, 
            # if we find a subset of size 8 with 8 distinct degrees, 
            # we need a subset of size 9 with 9 distinct degrees to beat it.
            # So we can't just break immediately unless best_count == n (impossible for induced subgraph of n vertices? max degree n-1).
            # Max distinct degrees is at most n-1.
            if best_count >= n - 1:
                break

    if best_mask == 0:
        # Fallback: return first min_k vertices
        return list(range(min_k))
        
    # Convert best_mask to list of vertex indices
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(v)
        m ^= lsb
        
    return result

# EVOLVE-BLOCK-END
