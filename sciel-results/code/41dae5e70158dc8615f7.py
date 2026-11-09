"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

# Precompute adjacency bitmask
_N = P['n']
_ADJ = [0] * _N
for _u, _v in P['edges']:
    _ADJ[_u] |= (1 << _v)
    _ADJ[_v] |= (1 << _u)

_MIN_VERTS = P['minimum_vertices']

def _distinct_degrees_for_mask(mask):
    """Count distinct degrees in the induced subgraph defined by mask."""
    degrees = set()
    for v in range(_N):
        if mask & (1 << v):
            deg = bin(_ADJ[v] & mask).count('1')
            degrees.add(deg)
    return len(degrees)

def _mask_to_list(mask):
    """Convert bitmask to list of vertex indices."""
    return [v for v in range(_N) if mask & (1 << v)]

# Precompute the best subset at module load time
# We do a full exhaustive search over all subsets of size >= 8
# This is feasible: sum(C(17,k) for k=8..17) = 65536 subsets total (since sum of all C(17,k) = 2^17 = 131072)
# Actually: C(17,8)=24310, C(17,9)=19448, C(17,10)=11440, C(17,11)=5740, C(17,12)=2380, C(17,13)=680, C(17,14)=136, C(17,15)=17, C(17,16)=1, C(17,17)=1
# Total = 24310+19448+11440+5740+2380+680+136+17+1+1 = 64153 subsets
# Each takes ~17 bit counts, so ~1.1M operations total. Very fast.

def _precompute_best():
    best_mask = 0
    best_count = 0
    # Iterate over all subsets of size >= 8
    for k in range(_MIN_VERTS, _N + 1):
        for subset in itertools.combinations(range(_N), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            cnt = _distinct_degrees_for_mask(mask)
            if cnt > best_count:
                best_count = cnt
                best_mask = mask
    return best_mask, best_count

_BEST_MASK, _BEST_COUNT = _precompute_best()
_BEST_LIST = _mask_to_list(_BEST_MASK)

def construct():
    # Return the precomputed best subset
    # Ensure we return a list of Python ints
    return list(_BEST_LIST)
# EVOLVE-BLOCK-END
