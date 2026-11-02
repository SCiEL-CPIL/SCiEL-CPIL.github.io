import numpy as np
import time

def construct():
    """
    Rank-56 tensor decomposition for 2x2 * 2x16 matrix multiplication.
    Uses Strassen's rank-7 decomposition per 2x2 column block, applied independently
    to 8 column blocks, exploiting the linearity of the output dimension.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 2, 2, 16
    num_blocks = p // m  # 8 column blocks of width 2

    # Strassen's rank-7 decomposition for 2x2 * 2x2
    # U: (4,7) with rows indexed by i*m+j = i*2+j
    # V: (4,7) with rows indexed by j*m+k = j*2+k  
    # W: (4,7) with rows indexed by k*n+i = k*2+i
    U_core = np.zeros((4, 7), dtype=np.int64)
    V_core = np.zeros((4, 7), dtype=np.int64)
    W_core = np.zeros((4, 7), dtype=np.int64)

    # m1 = (A00+A11)(B00+B11)
    U_core[0, 0] += 1
    U_core[3, 0] += 1
    V_core[0, 0] += 1
    V_core[3, 0] += 1
    W_core[0, 0] += 1
    W_core[3, 0] += 1

    # m2 = (A10+A11)B00
    U_core[2, 1] += 1
    U_core[3, 1] += 1
    V_core[0, 1] += 1
    W_core[2, 1] += 1
    W_core[3, 1] += -1

    # m3 = A00(B01-B11)
    U_core[0, 2] += 1
    V_core[1, 2] += 1
    V_core[3, 2] += -1
    W_core[1, 2] += 1
    W_core[3, 2] += 1

    # m4 = A11(B10-B00)
    U_core[3, 3] += 1
    V_core[2, 3] += 1
    V_core[0, 3] += -1
    W_core[0, 3] += 1
    W_core[2, 3] += 1

    # m5 = (A00+A01)B11
    U_core[0, 4] += 1
    U_core[1, 4] += 1
    V_core[3, 4] += 1
    W_core[0, 4] += -1
    W_core[1, 4] += 1

    # m6 = (A10-A00)(B00+B01)
    U_core[2, 5] += 1
    U_core[0, 5] += -1
    V_core[0, 5] += 1
    V_core[1, 5] += 1
    W_core[3, 5] += 1

    # m7 = (A01-A11)(B10+B11)
    U_core[1, 6] += 1
    U_core[3, 6] += -1
    V_core[2, 6] += 1
    V_core[3, 6] += 1
    W_core[0, 6] += 1

    # Expand to full 2x2x16 decomposition
    # U: (4, 7*8) = (4, 56)
    U = np.tile(U_core, (1, num_blocks))

    # V: (2*16, 56) = (32, 56)
    # Row index = j*p + k where j in {0,1}, k in {0,...,15}
    V = np.zeros((m * p, 7 * num_blocks), dtype=np.int64)
    for block in range(num_blocks):
        col_slice = slice(7 * block, 7 * block + 7)
        k0 = 2 * block
        k1 = 2 * block + 1
        # j=0: rows k0, k1
        V[k0, col_slice] = V_core[0, :]
        V[k1, col_slice] = V_core[1, :]
        # j=1: rows p+k0, p+k1
        V[p + k0, col_slice] = V_core[2, :]
        V[p + k1, col_slice] = V_core[3, :]

    # W: (16*2, 56) = (32, 56)
    # Row index = k*n + i where k in {0,...,15}, i in {0,1}
    W = np.zeros((p * n, 7 * num_blocks), dtype=np.int64)
    for block in range(num_blocks):
        col_slice = slice(7 * block, 7 * block + 7)
        k0 = 2 * block
        k1 = 2 * block + 1
        # k=k0: rows k0*2, k0*2+1
        W[k0 * n, col_slice] = W_core[0, :]
        W[k0 * n + 1, col_slice] = W_core[1, :]
        # k=k1: rows k1*2, k1*2+1
        W[k1 * n, col_slice] = W_core[2, :]
        W[k1 * n + 1, col_slice] = W_core[3, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
