"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
_START_TIME = time.time()
_DEADLINE = 120.0
_MARGIN = 5.0

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _build_rank88():
    """Build the known rank-88 decomposition."""
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: Schoolbook for A[0, :] (1x4) * B (4x8)
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: Strassen for A[1:3, :] (2x4) * B (4x8)
    strassen_terms = [
        {'U': {0: 1, 3: 1}, 'V': {0: 1, 3: 1}, 'W': {0: 1, 3: 1}},
        {'U': {2: 1, 3: 1}, 'V': {0: 1}, 'W': {2: 1, 3: -1}},
        {'U': {0: 1}, 'V': {1: 1, 3: -1}, 'W': {1: 1, 3: 1}},
        {'U': {3: 1}, 'V': {2: 1, 0: -1}, 'W': {0: 1, 2: 1}},
        {'U': {0: 1, 1: 1}, 'V': {3: 1}, 'W': {0: -1, 1: 1}},
        {'U': {2: 1, 0: -1}, 'V': {0: 1, 1: 1}, 'W': {3: 1}},
        {'U': {1: 1, 3: -1}, 'V': {2: 1, 3: 1}, 'W': {0: 1}}
    ]
    
    for j_block in range(4):
        # Block 1: A cols 0,1 * B rows 0,1
        for term in strassen_terms:
            for local_u, val in term['U'].items():
                lu, lv = divmod(local_u, 2)
                U[(1 + lu) * m + lv][col] += val
            for local_v, val in term['V'].items():
                lu, lv = divmod(local_v, 2)
                V[lu * p + (2 * j_block + lv)][col] += val
            for local_w, val in term['W'].items():
                lu, lv = divmod(local_w, 2)
                W[(2 * j_block + lv) * n + (1 + lu)][col] += val
            col += 1
            
        # Block 2: A cols 2,3 * B rows 2,3
        for term in strassen_terms:
            for local_u, val in term['U'].items():
                lu, lv = divmod(local_u, 2)
                U[(1 + lu) * m + (2 + lv)][col] += val
            for local_v, val in term['V'].items():
                lu, lv = divmod(local_v, 2)
                V[(2 + lu) * p + (2 * j_block + lv)][col] += val
            for local_w, val in term['W'].items():
                lu, lv = divmod(local_w, 2)
                W[(2 * j_block + lv) * n + (1 + lu)][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    return _build_rank88()
# EVOLVE-BLOCK-END
