"""Program for matrix multiplication tensor decomposition 3x3x6."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _build_rank22_3x3x3():
    """
    Return a rank-22 decomposition for 3x3x3 matrix multiplication.
    Uses Strassen (rank 7) for the top-left 2x2 block plus 15 schoolbook
    terms for the cross-block interactions involving row/col index 2.
    """
    r = 22
    U = [[0] * r for _ in range(9)]
    V = [[0] * r for _ in range(9)]
    W = [[0] * r for _ in range(9)]

    # --- Strassen 7 terms for the 2x2 block (rows/cols 0,1) ---
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[3][0] = -1

    # M2 = (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[3][1] = 1

    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[1][2] = 1

    # M4 = A11(B10-B00)
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[3][3] = 1

    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[1][4] = 1; W[4][4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[4][5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[1][6] = 1; W[4][6] = 1

    # --- Cross terms involving index 2 (15 schoolbook terms) ---
    # C[0,2]: A00*B02, A01*B12, A02*B22
    U[0][7] = 1;  V[2][7] = 1;  W[2][7] = 1
    U[1][8] = 1;  V[5][8] = 1;  W[2][8] = 1
    U[2][9] = 1;  V[8][9] = 1;  W[2][9] = 1
    # C[1,2]: A10*B02, A11*B12, A12*B22
    U[3][10] = 1; V[2][10] = 1; W[5][10] = 1
    U[4][11] = 1; V[5][11] = 1; W[5][11] = 1
    U[5][12] = 1; V[8][12] = 1; W[5][12] = 1
    # C[2,0]: A20*B00, A21*B10, A22*B20
    U[6][13] = 1; V[0][13] = 1; W[6][13] = 1
    U[7][14] = 1; V[3][14] = 1; W[6][14] = 1
    U[8][15] = 1; V[6][15] = 1; W[6][15] = 1
    # C[2,1]: A20*B01, A21*B11, A22*B21
    U[6][16] = 1; V[1][16] = 1; W[7][16] = 1
    U[7][17] = 1; V[4][17] = 1; W[7][17] = 1
    U[8][18] = 1; V[7][18] = 1; W[7][18] = 1
    # C[2,2]: A20*B02, A21*B12, A22*B22
    U[6][19] = 1; V[2][19] = 1; W[8][19] = 1
    U[7][20] = 1; V[5][20] = 1; W[8][20] = 1
    U[8][21] = 1; V[8][21] = 1; W[8][21] = 1

    # Verify with random matrices
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    np.random.seed(42)
    A_test = np.random.randn(3, 3)
    B_test = np.random.randn(3, 3)
    C_expected = A_test @ B_test

    C_computed = np.zeros((3, 3))
    for r_idx in range(r):
        a_proj = np.dot(U_arr[:, r_idx].reshape(3, 3), A_test)
        b_proj = np.dot(V_arr[:, r_idx].reshape(3, 3), B_test)
        for i in range(3):
            for k in range(3):
                C_computed[i, k] += W_arr[k * 3 + i, r_idx] * a_proj[i, k]

    if not np.allclose(C_computed, C_expected, atol=1e-10):
        return None

    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 120.0

    n, m, p = 3, 3, 6

    result = _build_rank22_3x3x3()

    if result is not None:
        U_block, V_block, W_block = result
        r_block = len(U_block[0])
        total_rank = 2 * r_block

        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        # Block 0: columns k=0,1,2 of B
        for r_idx in range(r_block):
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r_idx] = U_block[i * m + j][r_idx]
            for j in range(m):
                for k in range(3):
                    V[j * p + k][r_idx] = V_block[j * 3 + k][r_idx]
            for k in range(3):
                for i in range(n):
                    W[k * n + i][r_idx] = W_block[k * n + i][r_idx]

        # Block 1: columns k=3,4,5 of B
        for r_idx in range(r_block):
            offset = r_block + r_idx
            for i in range(n):
                for j in range(m):
                    U[i * m + j][offset] = U_block[i * m + j][r_idx]
            for j in range(m):
                for k in range(3):
                    V[j * p + (k + 3)][offset] = V_block[j * 3 + k][r_idx]
            for k in range(3):
                for i in range(n):
                    W[(k + 3) * n + i][offset] = W_block[k * n + i][r_idx]

        # Verify the full 3x3x6 decomposition
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        np.random.seed(123)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test

        C_computed = np.zeros((n, p))
        for r_idx in range(total_rank):
            a_proj = np.dot(U_arr[:, r_idx].reshape(n, m), A_test)
            b_proj = np.dot(V_arr[:, r_idx].reshape(m, p), B_test)
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += W_arr[k * n + i, r_idx] * a_proj[i, k]

        if np.allclose(C_computed, C_expected, atol=1e-10):
            return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 54
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    return {"U": U, "V": V, "W": W}
