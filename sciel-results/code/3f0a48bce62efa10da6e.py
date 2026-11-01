import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6

    # Rank 42 solution using Strassen blocks.
    # Decomposition: 2x4x6 is split into 2 (j-blocks) x 3 (k-blocks) = 6 sub-problems.
    # Each sub-problem is a 2x2x2 matrix multiplication, solved with Strassen (rank 7).
    # Total rank = 6 * 7 = 42.
    
    rank = 42

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # Strassen coefficients for 2x2x2
    # U1, V1, W1 are 4x7 matrices (flattened 2x2 indices)
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    strassen_U = [
        [1, 0, 1, 0, 1, 1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, -1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1],  # A11
    ]
    
    strassen_V = [
        [1, 1, 0, 0, 0, 1, 0],   # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, -1, 1, 0, 1], # B11
    ]
    
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0],  # C11
    ]

    col = 0
    for j_block in range(2):
        for k_block in range(3):
            for prod_idx in range(7):
                r = col + prod_idx

                for local_idx in range(4):
                    sign_u = strassen_U[local_idx][prod_idx]
                    if sign_u != 0:
                        gi = local_idx // 2
                        gj = local_idx % 2
                        U[gi * m + (j_block * 2 + gj), r] = sign_u

                    sign_v = strassen_V[local_idx][prod_idx]
                    if sign_v != 0:
                        gj = local_idx // 2
                        gk = local_idx % 2
                        V[(j_block * 2 + gj) * p + (k_block * 2 + gk), r] = sign_v

                    sign_w = strassen_W[local_idx][prod_idx]
                    if sign_w != 0:
                        gi = local_idx // 2
                        gk = local_idx % 2
                        W[(k_block * 2 + gk) * n + gi, r] = sign_w

            col += 7

    if time.time() - start_time > 110:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
