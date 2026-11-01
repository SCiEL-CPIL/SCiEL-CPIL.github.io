import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 exact tensor decomposition for 2x2 by 2x14 matrix multiplication.
    
    The tensor decomposes as the direct sum of 7 independent 2x2 x 2x2
    multiplication tensors. Each has minimal rank 7 (Strassen), so the
    total minimal rank is 7 * 7 = 49.
    
    Uses Strassen's 7-term algorithm per block with explicit coefficient tables.
    """
    _t0 = time.time()
    _DEADLINE = 115.0  # 35s safety margin below 150s kill

    n, m, p = DIMENSIONS  # 2, 2, 14
    num_blocks = p // 2  # 7 blocks
    rank = num_blocks * 7  # 49

    # Strassen terms: each is (a, b, c, d, e, f, g, h, C11, C12, C21, C22)
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h) -> C11 += 1, C22 += 1
    # M2 = (c+d)e     -> C21 += 1, C22 -= 1
    # M3 = a(f-h)     -> C12 += 1, C22 += 1
    # M4 = d(g-e)     -> C11 += 1, C21 += 1
    # M5 = (a+b)h     -> C11 -= 1, C12 += 1
    # M6 = (c-a)(e+f) -> C22 += 1
    # M7 = (b-d)(g+h) -> C11 += 1
    strassen_terms = [
        ( 1,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  1),  # M1
        ( 0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  1, -1),  # M2
        ( 1,  0,  0,  0,  0,  1,  0, -1,  0,  1,  0,  1),  # M3
        ( 0,  0,  0,  1, -1,  0,  1,  0,  1,  0,  1,  0),  # M4
        ( 1,  1,  0,  0,  0,  0,  0,  1, -1,  1,  0,  0),  # M5
        (-1,  0,  1,  0,  1,  1,  0,  0,  0,  0,  0,  1),  # M6
        ( 0,  1,  0, -1,  0,  0,  1,  1,  1,  0,  0,  0),  # M7
    ]

    # Pre-compute U rows (same for every block)
    U_rows = [
        [1,  0,  1,  0,  1, -1,  0],  # a
        [0,  0,  0,  0,  1,  0,  1],  # b
        [0,  1,  0,  0,  0,  1,  0],  # c
        [1,  1,  0,  1,  0,  0, -1],  # d
    ]

    # Pre-compute V local patterns for each term (indices into [e, f, g, h])
    V_local = [
        [1, 0, 0, 1],  # M1: e+h
        [1, 0, 0, 0],  # M2: e
        [0, 1, 0, -1], # M3: f-h
        [-1, 0, 1, 0], # M4: g-e
        [0, 0, 0, 1],  # M5: h
        [1, 1, 0, 0],  # M6: e+f
        [0, 0, 1, 1],  # M7: g+h
    ]

    # Pre-compute W local patterns for each term (indices into [C11, C12, C21, C22])
    W_local = [
        [1, 0, 0, 1],  # M1: C11+1, C22+1
        [0, 0, 1, -1], # M2: C21+1, C22-1
        [0, 1, 0, 1],  # M3: C12+1, C22+1
        [1, 0, 1, 0],  # M4: C11+1, C21+1
        [-1, 1, 0, 0], # M5: C11-1, C12+1
        [0, 0, 0, 1],  # M6: C22+1
        [1, 0, 0, 0],  # M7: C11+1
    ]

    # Build U: (n*m, rank) = (4, 49)
    # U[i, block*7 + s] = U_rows[i][s]
    U = np.zeros((n * m, rank), dtype=np.int64)
    for block in range(num_blocks):
        for s in range(7):
            col = block * 7 + s
            for i in range(n * m):
                U[i, col] = U_rows[i][s]

    # Build V: (m*p, rank) = (28, 49)
    # V[j*p + k, block*7 + s] where k = 2*block + local_col
    V = np.zeros((m * p, rank), dtype=np.int64)
    for block in range(num_blocks):
        for s in range(7):
            col = block * 7 + s
            for local_col in range(2):  # e/f (row 0) or g/h (row 1)
                for local_row in range(2):
                    v_idx = local_row * 2 + local_col  # 0=e,1=f,2=g,3=h
                    coeff = V_local[s][v_idx]
                    if coeff != 0:
                        row = local_row * p + 2 * block + local_col
                        V[row, col] = coeff

    # Build W: (p*n, rank) = (28, 49)
    # W[k*n + i, block*7 + s] where k = 2*block + local_col
    # C11 -> k=2*block, i=0; C12 -> k=2*block+1, i=0
    # C21 -> k=2*block, i=1; C22 -> k=2*block+1, i=1
    W = np.zeros((p * n, rank), dtype=np.int64)
    for block in range(num_blocks):
        for s in range(7):
            col = block * 7 + s
            # C11: k=2*block, i=0
            W[2 * block * n + 0, col] = W_local[s][0]
            # C12: k=2*block+1, i=0
            W[(2 * block + 1) * n + 0, col] = W_local[s][1]
            # C21: k=2*block, i=1
            W[2 * block * n + 1, col] = W_local[s][2]
            # C22: k=2*block+1, i=1
            W[(2 * block + 1) * n + 1, col] = W_local[s][3]

    if time.time() - _t0 > _DEADLINE:
        # Should never happen for this small fixed construction
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
