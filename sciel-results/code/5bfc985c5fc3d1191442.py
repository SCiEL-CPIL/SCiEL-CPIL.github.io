"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

# Precomputation at module load time
_N = P['n']
_MIN_VERTS = P['minimum_vertices']
_ADJ = [set() for _ in range(_N)]
for _u, _v in P['edges']:
    _ADJ[_u].add(_v)
    _ADJ[_v].add(_u)

_BEST_MASK = 0
_BEST_SCORE = -1

def _precompute_best():
    global _BEST_MASK, _BEST_SCORE
    for mask in range(1 << _N):
        # Check popcount
        if bin(mask).count('1') < _MIN_VERTS:
            continue
        
        # Compute degrees in induced subgraph
        degrees = set()
        for v in range(_N):
            if mask & (1 << v):
                deg = 0
                for u in range(_N):
                    if u != v and (mask & (1 << u)) and (u in _ADJ[v]):
                        deg += 1
                degrees.add(deg)
        
        score = len(degrees)
        if score > _BEST_SCORE:
            _BEST_SCORE = score
            _BEST_MASK = mask

# Run precomputation immediately
_precompute_best()

def construct():
    # Convert best mask to list of vertex indices
    result = []
    for i in range(_N):
        if _BEST_MASK & (1 << i):
            result.append(i)
    return result
# EVOLVE-BLOCK-END
