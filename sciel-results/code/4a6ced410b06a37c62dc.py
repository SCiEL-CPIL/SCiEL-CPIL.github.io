"""Precomputed lookup: best subset of the pinned 17-vertex graph maximizing distinct induced degrees."""
# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_edges = P['edges']
_adj = [0] * _N
for _a, _b in _edges:
    _adj[_a] |= (1 << _b)
    _adj[_b] |= (1 << _a)

# Precompute the optimal subset (size >= 8) maximizing distinct induced degrees.
# Enumerate all 2^17 = 131072 masks once at module load.
_best_mask = 0
_best_score = -1
for _mask in range(1 << _N):
    _pop = _mask.bit_count()
    if _pop < P['minimum_vertices']:
        continue
    # Compute degrees of each vertex in the induced subgraph
    _degs = []
    _m = _mask
    while _m:
        _lsb = _m & (-_m)
        _v = _lsb.bit_length() - 1
        _degs.append((_adj[_v] & _mask).bit_count())
        _m ^= _lsb
    _distinct = len(set(_degs))
    if _distinct > _best_score:
        _best_score = _distinct
        _best_mask = _mask

# Convert the winning mask to a list of vertex indices (Python ints).
_best_vertices = [_i for _i in range(_N) if (_best_mask >> _i) & 1]

def construct():
    return list(_best_vertices)
# EVOLVE-BLOCK-END
