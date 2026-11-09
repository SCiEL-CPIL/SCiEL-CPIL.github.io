"""Rewritten program: precomputed lookup for max distinct degrees."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
# Precompute adjacency as bitmask integers
_adj_mask = [0] * _N
for (u, v) in P['edges']:
    _adj_mask[u] |= (1 << v)
    _adj_mask[v] |= (1 << u)

def _popcount(x):
    """Count bits in integer x."""
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c

def _distinct_degrees_mask(mask):
    """Count distinct degrees in induced subgraph on vertices given by bitmask `mask`."""
    degs = set()
    for v in range(_N):
        if mask & (1 << v):
            degs.add(_popcount(_adj_mask[v] & mask))
    return len(degs)

# Precompute the best mask and score at module load time
_total_masks = 1 << _N
_best_mask = 0
_best_score = -1

for _m in range(_total_masks):
    if _popcount(_m) < 8:
        continue
    _s = _distinct_degrees_mask(_m)
    if _s > _best_score:
        _best_score = _s
        _best_mask = _m

def construct():
    result = []
    for v in range(_N):
        if _best_mask & (1 << v):
            result.append(v)
    return result
# EVOLVE-BLOCK-END
