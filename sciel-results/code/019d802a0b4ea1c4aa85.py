# EVOLVE-BLOCK-START
import time

_P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

_N = _P['n']
_EDGES = _P['edges']

# adjacency bitmasks
_ADJ = [0] * _N
for _a, _b in _EDGES:
    _ADJ[_a] |= (1 << _b)
    _ADJ[_b] |= (1 << _a)

def _is_clique_mask(mask):
    while mask:
        v = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << v)
        if (_ADJ[v] & rest) != rest:
            return False
        mask = rest
    return True

def _is_indep_mask(mask):
    while mask:
        v = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << v)
        if (_ADJ[v] & rest) != 0:
            return False
        mask = rest
    return True

# precompute clique/indep for all masks
_CLIQ = [False] * (1 << _N)
_INDEP = [False] * (1 << _N)
for _m in range(1 << _N):
    if _m == 0:
        _CLIQ[_m] = True
        _INDEP[_m] = True
        continue
    _CLIQ[_m] = _is_clique_mask(_m)
    _INDEP[_m] = _is_indep_mask(_m)

# Precompute: for each mask, can it be covered by k parts where each part is clique or indep?
_VALID_PART = [False] * (1 << _N)
for _m in range(1 << _N):
    if _CLIQ[_m] or _INDEP[_m]:
        _VALID_PART[_m] = True

# Precompute minimum parts to cover each subset (DP over subsets)
# min_parts[mask] = minimum number of valid parts that partition mask
_MIN_PARTS = [0] * (1 << _N)
# Store the best submask choice for reconstruction
_BEST_SUBMASK = [0] * (1 << _N)
for _m in range(1, 1 << _N):
    _MIN_PARTS[_m] = 10**9

for _m in range(1, 1 << _N):
    best = 10**9
    best_s = 0
    lowbit = _m & (-_m)
    s = _m
    while s:
        if (s & lowbit) and _VALID_PART[s]:
            rest = _m ^ s
            val = _MIN_PARTS[rest] + 1
            if val < best:
                best = val
                best_s = s
        s = (s - 1) & _m
    _MIN_PARTS[_m] = best
    _BEST_SUBMASK[_m] = best_s

def construct():
    full = (1 << _N) - 1
    parts = []
    mask = full
    while mask:
        s = _BEST_SUBMASK[mask]
        if s == 0:
            # fallback: singleton
            s = mask & (-mask)
        parts.append([x for x in range(_N) if (s >> x) & 1])
        mask ^= s
    return parts
# EVOLVE-BLOCK-END
