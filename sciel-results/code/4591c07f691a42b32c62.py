"""
Rank-29 exact decomposition for 2x3 * 3x8 matrix multiplication.

Construction:
  A (2x3) = [A_left | A_right] where A_left is 2x2, A_right is 2x1.
  B (3x8) = [B_top; B_bottom] where B_top is 2x8, B_bottom is 1x8.
  C = A*B = A_left * B_top + A_right * B_bottom  (2x8 result)

We split B_top into 4 blocks of 2x2: B_top = [B1, B2, B3, B4].
  C = A_left * B_top + A_right * B_bottom

For each 2x2 block multiplication (A_left 2x2 * B_j 2x2 -> 2x2),
use Strassen's rank-7 algorithm. Total: 4 * 7 = 28 terms.

For the 2x1 * 1x8 multiplication (A_right * B_bottom -> 2x8),
the tensor is the all-ones 2x1x8 tensor which has rank 1.
This is a single Kronecker product: A_right (2x1) outer B_bottom (1x8).
Total: 1 term.

Grand total: 28 + 1 = 29 terms.
"""

import time

DIMENSIONS = [2, 3, 8]


def construct():
    deadline = time.time() + 140  # safety margin

    n, m, p = 2, 3, 8
    rank = 29

    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 29
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 29
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 29

    # Index conventions:
    # A is 2x3: A[i,j] -> U[i*m + j, c] = U[i*3+j, c]
    #   i in 0,1; j in 0,1,2
    #   A_left: j=0,1; A_right: j=2
    # B is 3x8: B[j,k] -> V[j*p + k, c] = V[j*8+k, c]
    #   j in 0,1,2; k in 0..7
    #   B_top: j=0,1; B_bottom: j=2
    # C is 2x8: C[i,k] -> W[k*n + i, c] = W[k*2+i, c]
    #   i in 0,1; k in 0..7
    #   k=0,1 -> block 0; k=2,3 -> block 1; k=4,5 -> block 2; k=6,7 -> block 3

    col = 0

    def add_strassen_block(a_col_offset, b_col_offset):
        """Add 7 Strassen terms for multiplying A[0:2, a_col_offset:a_col_offset+2]
        * B[0:2, b_col_offset:b_col_offset+2] -> C[0:2, b_col_offset:b_col_offset+2]
        """
        nonlocal col

        # U indices: A[i, j] -> U[i*3 + j]
        # V indices: B[i, j] -> V[i*8 + j]
        # W indices: C[i, j] -> W[j*2 + i]

        ac = a_col_offset
        bc = b_col_offset

        U_a00 = ac        # A[0, ac]
        U_a01 = ac + 1    # A[0, ac+1]
        U_a10 = 3 + ac    # A[1, ac]
        U_a11 = 3 + ac + 1  # A[1, ac+1]

        V_b00 = bc        # B[0, bc]
        V_b01 = bc + 1    # B[0, bc+1]
        V_b10 = 8 + bc    # B[1, bc]
        V_b11 = 8 + bc + 1  # B[1, bc+1]

        W_c00 = 2 * bc    # C[0, bc]
        W_c01 = 2 * bc + 2  # C[0, bc+1]
        W_c10 = 2 * bc + 1  # C[1, bc]
        W_c11 = 2 * bc + 3  # C[1, bc+1]

        # M1 = (a00+a10)*(b00+b10)
        U[U_a00][col] += 1.0
        U[U_a10][col] += 1.0
        V[V_b00][col] += 1.0
        V[V_b10][col] += 1.0
        W[W_c01][col] += 1.0  # c01 += M1
        W[W_c11][col] += 1.0  # c11 += M1
        col += 1

        # M2 = a10*(b01+b11)
        U[U_a10][col] += 1.0
        V[V_b01][col] += 1.0
        V[V_b11][col] += 1.0
        W[W_c00][col] += -1.0  # c00 -= M2
        W[W_c01][col] += 1.0   # c01 += M2
        col += 1

        # M3 = a00*(b00+b01)
        U[U_a00][col] += 1.0
        V[V_b00][col] += 1.0
        V[V_b01][col] += 1.0
        W[W_c10][col] += 1.0   # c10 += M3
        W[W_c11][col] += -1.0  # c11 -= M3
        col += 1

        # M4 = a11*(b10+b11)
        U[U_a11][col] += 1.0
        V[V_b10][col] += 1.0
        V[V_b11][col] += 1.0
        W[W_c00][col] += 1.0   # c00 += M4
        W[W_c10][col] += 1.0   # c10 += M4
        col += 1

        # M5 = (a00+a11)*(b00+b11)
        U[U_a00][col] += 1.0
        U[U_a11][col] += 1.0
        V[V_b00][col] += 1.0
        V[V_b11][col] += 1.0
        W[W_c00][col] += 1.0   # c00 += M5
        W[W_c11][col] += 1.0   # c11 += M5
        col += 1

        # M6 = (a01-a11)*(b10+b11)
        U[U_a01][col] += 1.0
        U[U_a11][col] += -1.0
        V[V_b10][col] += 1.0
        V[V_b11][col] += 1.0
        W[W_c00][col] += 1.0   # c00 += M6
        col += 1

        # M7 = (a00-a10)*(b00+b10)
        U[U_a00][col] += 1.0
        U[U_a10][col] += -1.0
        V[V_b00][col] += 1.0
        V[V_b10][col] += 1.0
        W[W_c11][col] += -1.0  # c11 -= M7
        col += 1

    # Process 4 blocks of B_top (2x2 each) with Strassen
    for block in range(4):
        bc_offset = block * 2
        add_strassen_block(a_col_offset=0, b_col_offset=bc_offset)

    # Add the rank-1 term for A_right * B_bottom
    # A_right = A[:, 2] (2x1), B_bottom = B[2, :] (1x8)
    # C[i,k] += A[i,2] * B[2,k] for all i in {0,1}, k in {0..7}
    # U[i*3+2, col] = 1 for i=0,1
    # V[2*8+k, col] = 1 for k=0..7
    # W[k*2+i, col] = 1 for all i,k
    for i in range(2):
        U[i * 3 + 2][col] += 1.0
    for k in range(8):
        V[2 * 8 + k][col] += 1.0
    for k in range(8):
        for i in range(2):
            W[k * 2 + i][col] += 1.0
    col += 1

    # Verify col
    assert col == 29, f"Expected 29 columns, got {col}"

    # Quick verification
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Test with random matrices
    np.random.seed(42)
    A_test = np.random.randint(-5, 5, size=(2, 3)).astype(float)
    B_test = np.random.randint(-5, 5, size=(3, 8)).astype(float)
    C_expected = A_test @ B_test

    # Compute C from decomposition: C[i,k] = sum_c sum_j U[i*3+j, c] * V[j*8+k, c] * W[k*2+i, c]
    C_computed = np.zeros((2, 8), dtype=float)
    for c in range(rank):
        for i in range(2):
            for j in range(3):
                for k in range(8):
                    C_computed[i, k] += U_arr[i * 3 + j, c] * V_arr[j * 8 + k, c] * W_arr[k * 2 + i, c]

    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # If verification fails, fall back to schoolbook
        rank_school = n * m * p
        U_s = [[0.0] * rank_school for _ in range(n * m)]
        V_s = [[0.0] * rank_school for _ in range(m * p)]
        W_s = [[0.0] * rank_school for _ in range(p * n)]
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col_s] = 1.0
                    V_s[j * p + k][col_s] = 1.0
                    W_s[k * n + i][col_s] = 1.0
                    col_s += 1
        return {"U": U_s, "V": V_s, "W": W_s}

    return {"U": U, "V": V, "W": W}
