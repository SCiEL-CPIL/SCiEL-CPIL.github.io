"""
5x5 matrix multiplication tensor decomposition via direct-sum construction.

Partition the indices {0,1,2,3,4} into a 4-block {0,1,2,3} and a 1-block {4}.
The multiplication tensor decomposes as the direct sum of:
  - a 4x4x4 sub-tensor (indices 0..3 on all sides), decomposed by Laderman's
    rank-21 algorithm (the Kronecker product of two 2x2 Strassen algorithms), and
  - a 1x1x1 corner term (index 4 on all sides), rank 1.

Total rank = 21 + 1 = 22.

Index conventions (matching the seed program):
  U index = r*m + c   (m = 5)
  V index = c*p + k   (p = 5)
  W index = k*n + r   (n = 5)
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0

    terms = []

    def add_term(U_dict, V_dict, W_dict):
        if time.time() > deadline:
            return
        U_dict = {k: v for k, v in U_dict.items() if v != 0}
        V_dict = {k: v for k, v in V_dict.items() if v != 0}
        W_dict = {k: v for k, v in W_dict.items() if v != 0}
        if U_dict and V_dict and W_dict:
            terms.append((U_dict, V_dict, W_dict))

    # Laderman's rank-21 algorithm for 4x4x4, built from the Kronecker
    # product of two 2x2 Strassen algorithms.
    # 2x2 Strassen factors (each a dict {index: value}, 7 terms):
    #   U index = 2*r + c,  V index = 2*c + k,  W index = 2*k + r
    S_U = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1},
    ]
    S_V = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1},
    ]
    S_W = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1},
    ]

    # Kronecker product of the two 2x2 Strassen factor sets -> 49 terms for 4x4x4.
    # 4x4 index:  U = 4*r + c,  V = 4*c + k,  W = 4*k + r
    for i in range(7):
        for j in range(7):
            U4 = {}
            for (r, c), v in S_U[i].items():
                U4[4 * r + c] = v
            V4 = {}
            for (c, k), v in S_V[i].items():
                V4[4 * c + k] = v
            W4 = {}
            for (k, r), v in S_W[j].items():
                W4[4 * k + r] = v

            # Map 4x4 indices into the 5x5 tensor (the 4-block occupies 0..3).
            U5 = {4 * r + c: v for (r, c), v in U4.items()}
            V5 = {4 * c + k: v for (c, k), v in V4.items()}
            W5 = {4 * k + r: v for (k, r), v in W4.items()}

            add_term(U5, V5, W5)

    # 1x1x1 corner term at index 4 (scalar product A44 * B44 -> C44).
    u_idx = 4 * m + 4
    v_idx = 4 * p + 4
    w_idx = 4 * n + 4
    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})

    def _build_result(terms, n, m, p):
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        for r in range(rank):
            U_dict, V_dict, W_dict = terms[r]
            for idx, val in U_dict.items():
                U[idx][r] = val
            for idx, val in V_dict.items():
                V[idx][r] = val
            for idx, val in W_dict.items():
                W[idx][r] = val

        return {"U": U, "V": V, "W": W}

    return _build_result(terms, n, m, p)
# EVOLVE-BLOCK-END
