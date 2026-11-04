import numpy as np
import time

DIMENSIONS = [4, 4, 6]

_STRassen_U = np.array([
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1]
], dtype=float)

_STRassen_V = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1]
], dtype=float)

_STRassen_W = np.array([
    [1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0, 1, 0]
], dtype=float)

def construct():
    t0 = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Part 1: 4x4 * 4x4
    # Recursive Strassen
    # Outer Strassen coeffs
    outer_U = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    outer_V = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    outer_W = [
        {0: 1, 3: 1},
        {1: 1},
        {0: -1},
        {1: 1},
        {3: 1},
        {2: 1},
        {0: 1, 2: -1}
    ]
    
    # Block offsets for W in 4x4 case (p*n = 16)
    # C00: k=0,1; i=0,1 -> 0,1,4,5
    # C01: k=0,1; i=2,3 -> 2,3,6,7
    # C10: k=2,3; i=0,1 -> 8,9,12,13
    # C11: k=2,3; i=2,3 -> 10,11,14,15
    w_offsets_444 = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # U
            for block_idx, coeff in outer_U[s].items():
                offset = block_idx * 4
                U[offset:offset+4, col] += coeff * _STRassen_U[:, t]
            
            # V
            for block_idx, coeff in outer_V[s].items():
                offset = block_idx * 4
                V[offset:offset+4, col] += coeff * _STRassen_V[:, t]
            
            # W
            for block_idx, coeff in outer_W[s].items():
                w_offs = w_offsets_444[block_idx]
                for local_idx, w_off in enumerate(w_offs):
                    W[w_off, col] += coeff * _STRassen_W[local_idx, t]

    # Part 2: 4x4 * 4x2
    # Blocks:
    # A00 (0-3) * B_top (4,5,10,11) -> C_top (16,17,20,21)
    # A01 (4-7) * B_bot (16,17,22,23) -> C_top (16,17,20,21)
    # A10 (8-11) * B_top (4,5,10,11) -> C_bot (18,19,22,23)
    # A11 (12-15) * B_bot (16,17,22,23) -> C_bot (18,19,22,23)
    
    blocks_442 = [
        (0, [4, 5, 10, 11], [16, 17, 20, 21]),
        (1, [16, 17, 22, 23], [16, 17, 20, 21]),
        (2, [4, 5, 10, 11], [18, 19, 22, 23]),
        (3, [16, 17, 22, 23], [18, 19, 22, 23])
    ]
    
    for bi, (a_block, v_offs, w_offs) in enumerate(blocks_442):
        for t in range(7):
            col = rank_444 + bi * 7 + t
            
            a_offset = a_block * 4
            U[a_offset:a_offset+4, col] = _STRassen_U[:, t]
            
            for local_idx, v_off in enumerate(v_offs):
                V[v_off, col] = _STRassen_V[local_idx, t]
            
            for local_idx, w_off in enumerate(w_offs):
                W[w_off, col] = _STRassen_W[local_idx, t]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
