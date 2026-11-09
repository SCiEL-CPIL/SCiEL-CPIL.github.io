import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_adj_mask = [0] * _N
for (u, v) in P['edges']:
    _adj_mask[u] |= (1 << v)
    _adj_mask[v] |= (1 << u)

# Precompute popcount for all 17-bit masks
_popcount_table = [0] * (1 << _N)
for _i in range(1, 1 << _N):
    _popcount_table[_i] = _popcount_table[_i >> 1] + (_i & 1)

def _popcount(x):
    return _popcount_table[x]

def _distinct_degrees_for_mask(mask):
    degs = set()
    m = mask
    while m:
        lb = m & (-m)
        v = lb.bit_length() - 1
        d = _popcount(_adj_mask[v] & mask)
        degs.add(d)
        m ^= lb
    return len(degs)

def _next_combination(x):
    u = x & -x
    v = u + x
    return v + (((v ^ x) // u) >> 2)

def construct():
    start = time.time()
    deadline = start + 85.0
    
    best_score = 0
    best_mask = 0
    
    # Iterate subset sizes from largest to smallest
    for k in range(_N, 7, -1):
        # If we already have a score >= k, no subset of size k or smaller can beat it
        if best_score >= k:
            break
            
        # Generate all masks with exactly k bits set
        # Start with the smallest mask with k bits: (1 << k) - 1
        mask = (1 << k) - 1
        limit = 1 << _N
        
        while mask < limit:
            if time.time() > deadline:
                # Return best found so far
                break
                
            score = _distinct_degrees_for_mask(mask)
            if score > best_score:
                best_score = score
                best_mask = mask
                # If we achieved the maximum possible for this size k, 
                # we can't do better for this k, and since we are decreasing k,
                # we can break the inner loop. The outer loop check will handle the rest.
                if score == k:
                    break
            
            mask = _next_combination(mask)
            
        if time.time() > deadline:
            break

    # Convert best_mask to list of Python ints
    result = []
    m = best_mask
    while m:
        lb = m & (-m)
        v = lb.bit_length() - 1
        result.append(int(v))
        m ^= lb
    result.sort()
    return result
