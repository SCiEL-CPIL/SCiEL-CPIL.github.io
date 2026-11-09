"""
Optimized exhaustive search for the maximum number of distinct degrees
in an induced subgraph of at least 8 vertices from a 17-vertex graph.
"""
# EVOLVE-BLOCK-START
import time

# Fixed graph parameters
P = {
    'n': 17,
    'edges': [
        [0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16],
        [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16],
        [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15],
        [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16],
        [4, 5], [4, 6], [4, 8], [4, 12], [4, 13],
        [5, 6], [5, 7], [5, 9], [5, 13], [5, 14],
        [6, 7], [6, 8], [6, 10], [6, 14], [6, 15],
        [7, 8], [7, 9], [7, 11], [7, 15], [7, 16],
        [8, 9], [8, 10], [8, 12], [8, 16],
        [9, 10], [9, 11], [9, 13],
        [10, 11], [10, 12], [10, 14],
        [11, 12], [11, 13], [11, 15],
        [12, 13], [12, 14], [12, 16],
        [13, 14], [13, 15],
        [14, 15], [14, 16],
        [15, 16]
    ],
    'homogeneous_cap': 4,
    'minimum_vertices': 8
}

_N = P['n']
_MIN_VERTS = P['minimum_vertices']

# Precompute adjacency masks for each vertex
_adj_mask = [0] * _N
for u, v in P['edges']:
    _adj_mask[u] |= (1 << v)
    _adj_mask[v] |= (1 << u)

# Precompute popcount for all 17-bit masks
# Using a lookup table for speed, or built-in bit_count() if Python >= 3.10
# Assuming Python 3.8+ for compatibility, we use a helper or precomputed list.
# A list of size 2^17 is 131,072 entries, which is small.
_MAX_MASK = 1 << _N
_popcount = [0] * _MAX_MASK
for i in range(1, _MAX_MASK):
    _popcount[i] = _popcount[i >> 1] + (i & 1)

# Precompute the best mask and its score
_best_mask = 0
_best_score = 0

# Iterate over all possible masks
# We can skip masks with popcount < _MIN_VERTS
for mask in range(_MAX_MASK):
    if _popcount[mask] < _MIN_VERTS:
        continue
    
    # Calculate distinct degrees for this mask
    deg_bits = 0
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        # Degree of v in the induced subgraph is the number of neighbors of v also in mask
        deg = _popcount[_adj_mask[v] & mask]
        deg_bits |= (1 << deg)
        m ^= lsb
    
    # Count distinct degrees
    score = deg_bits.bit_count()
    
    if score > _best_score:
        _best_score = score
        _best_mask = mask

def construct():
    """
    Returns the list of vertices corresponding to the best mask found.
    The list contains at least 8 distinct integers.
    """
    result = []
    m = _best_mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        result.append(int(v))
        m ^= lsb
    result.sort()
    return result
# EVOLVE-BLOCK-END
