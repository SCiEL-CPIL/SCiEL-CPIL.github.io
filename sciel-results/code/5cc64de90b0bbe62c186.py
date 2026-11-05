import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Strategy: 2x2 blocks of 3x3 using Strassen (rank 7) on the block level.
    # Total rank = 27 * 7 = 189.
    # This is a Kronecker product decomposition: T_6 = T_2 ⊗ T_3.
    # We construct the factors explicitly for efficiency and correctness.

    rank = 189
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)

    # Strassen 2x2 decomposition coefficients
    # r=0: (A11+A22)(B11+B22) -> C11+, C22+
    # r=1: (A21+A22)B11 -> C21+, C22-
    # r=2: A11(B12-B22) -> C12+, C22+
    # r-3: A22(B21-B11) -> C11+, C21+
    # r=4: (A11+A12)B22 -> C11-, C12+
    # r=5: (A21-A11)(B11+B12) -> C22+
    # r=6: (A12-A22)(B21+B22) -> C11+

    # For each of the 27 block positions (i, j, k) where i,k are row/col of A, j is col of B/row of C
    # We have a 2x2 sub-block for A (rows 2i, 2i+1; cols 2k, 2k+1)
    # A 2x2 sub-block for B (rows 2k, 2k+1; cols 2j, 2j+1)
    # A 2x2 sub-block for C (rows 2j, 2j+1; cols 2i, 2i+1)

    # Strassen terms definition:
    # Each term s in 0..6 has:
    # U coefficients: dict mapping (row_in_2x2, col_in_2x2) -> coeff
    # V coefficients: dict mapping (row_in_2x2, col_in_2x2) -> coeff
    # W coefficients: dict mapping (row_in_2x2, col_in_2x2) -> coeff
    # Note: W is indexed as W[k*n + i, r]. In the 2x2 block, the local indices for C are (c_row, c_col) which correspond to (k_local, i_local).
    # So W local index is c_row * 2 + c_col.

    strassen_U = [
        {(0,0): 1, (1,1): 1},      # r=0
        {(1,0): 1, (1,1): 1},      # r=1
        {(0,0): 1},                # r=2
        {(1,1): 1},                # r=3
        {(0,0): 1, (0,1): 1},      # r=4
        {(1,0): 1, (0,0): -1},     # r=5
        {(0,1): 1, (1,1): -1},     # r=6
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},      # r=0
        {(0,0): 1},                # r=1
        {(0,1): 1, (1,1): -1},     # r=2
        {(1,0): 1, (0,0): -1},     # r=3
        {(1,1): 1},                # r=4
        {(0,0): 1, (0,1): 1},      # r=5
        {(1,0): 1, (1,1): 1},      # r=6
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},      # r=0
        {(1,0): 1, (1,1): -1},     # r=1
        {(0,1): 1, (1,1): 1},      # r=2
        {(0,0): 1, (1,0): 1},      # r=3
        {(0,0): -1, (0,1): 1},     # r=4
        {(1,1): 1},                # r=5
        {(0,0): 1},                # r=6
    ]

    idx = 0
    for i in range(3):  # block row of A
        for k in range(3):  # block col of A / block row of B
            for j in range(3):  # block col of B / block row of C
                for s in range(7):
                    # Fill U
                    for (u_r, u_c), val_u in strassen_U[s].items():
                        global_i = 2 * i + u_r
                        global_j = 2 * k + u_c
                        U[global_i * 6 + global_j, idx] += val_u

                    # Fill V
                    for (v_r, v_c), val_v in strassen_V[s].items():
                        global_j = 2 * k + v_r
                        global_k = 2 * j + v_c
                        V[global_j * 6 + global_k, idx] += val_v

                    # Fill W
                    # W index is k * n + i.
                    # Local k is the row of C block, local i is the col of C block.
                    for (w_r, w_c), val_w in strassen_W[s].items():
                        global_k = 2 * j + w_r
                        global_i = 2 * i + w_c
                        W[global_k * 6 + global_i, idx] += val_w

                    idx += 1

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verification (lightweight)
    def verify(U, V, W, rank, num_samples=50):
        np.random.seed(42)
        for _ in range(num_samples):
            i = np.random.randint(0, 6)
            j = np.random.randint(0, 6)
            k = np.random.randint(0, 6)
            s = 0.0
            for r in range(rank):
                s += U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
            if abs(s - 1.0) > 1e-6:
                return False
        return True

    if not verify(U, V, W, rank):
        # Fallback to schoolbook if something went wrong (shouldn't happen)
        rank = 216
        U = np.zeros((36, rank), dtype=int)
        V = np.zeros((36, rank), dtype=int)
        W = np.zeros((36, rank), dtype=int)
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    r = i * 36 + j * 6 + k
                    U[i * 6 + j, r] = 1
                    V[j * 6 + k, r] = 1
                    W[k * 6 + i, r] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
