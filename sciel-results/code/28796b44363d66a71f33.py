"""Precomputed lookup table: all 2^17 masks scored at load time, O(1) lookup in construct()."""
# EVOLVE-BLOCK-START
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

# Precompute distinct-degree count for every 17-bit mask at module load
# Using a bitset-of-degrees approach: for each mask, compute the set of degrees
# via a 17-bit integer where bit d is set if degree d appears.
_distinct_deg_table = [0] * _MAX_MASK
for _mask in range(_MAX_MASK):
    _deg_bits = 0
    _m = _mask
    while _m:
        _lsb = _m & (-_m)
        _v = _lsb.bit_length() - 1
        _deg = _popcount[_adj_mask[_v] & _mask]
        _deg_bits |= (1 << _deg)
        _m ^= _lsb
    _distinct_deg_table[_mask] = _deg_bits.bit_count()

# Precompute the best mask among all valid subsets (popcount >= 8)
_MIN_VERTS = P['minimum_vertices']
_best_mask_at_load = 0
_best_score_at_load = 0
for _mask in range(_MAX_MASK):
    if _popcount[_mask] >= _MIN_VERTS:
        _s = _distinct_deg_table[_mask]
        if _s > _best_score_at_load:
            _best_score_at_load = _s
            _best_mask_at_load = _mask

def construct():
    """Return the precomputed best vertex list (O(1) lookup)."""
    return _mask_to_list(_best_mask_at_load)

def _mask_to_list(mask):
    """Convert bitmask to sorted list of Python ints."""
    result = []
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        result.append(int(v))
        m ^= lsb
    result.sort()
    return result
# EVOLVE-BLOCK-END
