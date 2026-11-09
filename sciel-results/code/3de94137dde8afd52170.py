"""Combined program: bitmask enumeration with popcount table + early-termination pruning."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_adj_mask = [0] * _N
for (u, v) in P['edges']:
    _adj_mask[u] |= (1 << v)
    _adj_mask[v] |= (1 << u)

# Precompute popcount for all 17-bit masks
_MAX_MASK = 1 << _N
_popcount = [0] * _MAX_MASK
for _i in range(1, _MAX_MASK):
    _popcount[_i] = _popcount[_i >> 1] + (_i & 1)

def _distinct_degrees_mask(mask):
    """Count distinct degrees in induced subgraph represented by bitmask."""
    degs = set()
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        deg = _popcount[_adj_mask[v] & mask]
        degs.add(deg)
        m ^= lsb
    return len(degs)

def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin under 90s kill
    best_score = -1
    best_mask = (1 << 8) - 1  # fallback: vertices 0-7

    n = _N
    # Enumerate subsets by size, starting from 8
    for r in range(8, n + 1):
        for comb in itertools.combinations(range(n), r):
            if time.time() > deadline:
                break
            mask = 0
            for v in comb:
                mask |= (1 << v)
            score = _distinct_degrees_mask(mask)
            if score > best_score:
                best_score = score
                best_mask = mask
        if time.time() > deadline:
            break

    # Convert bitmask to sorted list of Python ints
    result = []
    m = best_mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        result.append(int(v))
        m ^= lsb
    result.sort()
    return result
# EVOLVE-BLOCK-END
