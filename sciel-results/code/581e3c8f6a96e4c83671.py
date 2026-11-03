"""Matrix multiplication 4,5,5 rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_4x4x4():
    """
    Returns (U, V, W) for 4x4x4 matrix multiplication using Strassen's algorithm.
    Indexing:
      U: (4*4, rank) = (16, rank), U[i*4+j, c]
      V: (4*4, rank) = (16, rank), V[j*4+k, c]
      W: (4*4, rank) = (16, rank), W[k*4+i, c]
    """
    n = 4
    rank = 7
    U = np.zeros((n * n, rank), dtype=float)
    V = np.zeros((n * n, rank), dtype=float)
    W = np.zeros((n * n, rank), dtype=float)

    # Strassen for 2x2x2 with 7 multiplications, embedded in 4x4x4
    # We split each 4x4 matrix into four 2x2 blocks:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]], C = [[C11, C12], [C21, C22]]
    # Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For each of the 7 multiplications, we need to define:
    # - which entries of U (A-part) are non-zero and their values
    # - which entries of V (B-part) are non-zero and their values
    # - which entries of W (C-part) are non-zero and their values

    # Helper to get global index for a 4x4 matrix entry (row, col)
    def a_idx(r, c):  # A: row i, col j
        return r * 4 + c
    def b_idx(r, c):  # B: row j, col k
        return r * 4 + c
    def c_idx(r, c):  # C: row k? No, W is indexed as W[k*n+i] for C[i,k]
        # C[i, k] -> W[k*4 + i]
        return c * 4 + r  # r=i, c=k

    # Block boundaries: 0,1 are top/left; 2,3 are bottom/right
    top_left = [(0, 0), (0, 1), (1, 0), (1, 1)]
    top_right = [(0, 2), (0, 3), (1, 2), (1, 3)]
    bottom_left = [(2, 0), (2, 1), (3, 0), (3, 1)]
    bottom_right = [(2, 2), (2, 3), (3, 2), (3, 3)]

    def set_block(matrix, rows_cols, value, col_idx):
        for (r, c) in rows_cols:
            matrix[r * 4 + c, col_idx] = value

    # M1 = (A11 + A22)(B11 + B22)
    # A-part: A11 + A22
    set_block(U, top_left, 1, 0)
    set_block(U, bottom_right, 1, 0)
    # B-part: B11 + B22
    set_block(V, top_left, 1, 0)
    set_block(V, bottom_right, 1, 0)
    # C-part: contributes to C11 (+1) and C22 (+1)
    set_block(W, top_left, 1, 0)  # C11
    set_block(W, bottom_right, 1, 0)  # C22

    # M2 = (A21 + A22)B11
    # A-part: A21 + A22
    set_block(U, bottom_left, 1, 1)
    set_block(U, bottom_right, 1, 1)
    # B-part: B11
    set_block(V, top_left, 1, 1)
    # C-part: contributes to C21 (+1) and C22 (-1)
    set_block(W, bottom_left, 1, 1)  # C21
    set_block(W, bottom_right, -1, 1)  # C22

    # M3 = A11(B12 - B22)
    # A-part: A11
    set_block(U, top_left, 1, 2)
    # B-part: B12 - B22
    set_block(V, top_right, 1, 2)
    set_block(V, bottom_right, -1, 2)
    # C-part: contributes to C12 (+1) and C22 (+1)
    set_block(W, top_right, 1, 2)  # C12
    set_block(W, bottom_right, 1, 2)  # C22

    # M4 = A22(B21 - B11)
    # A-part: A22
    set_block(U, bottom_right, 1, 3)
    # B-part: B21 - B11
    set_block(V, bottom_left, 1, 3)
    set_block(V, top_left, -1, 3)
    # C-part: contributes to C11 (+1) and C21 (+1)
    set_block(W, top_left, 1, 3)  # C11
    set_block(W, bottom_left, 1, 3)  # C21

    # M5 = (A11 + A12)B22
    # A-part: A11 + A12
    set_block(U, top_left, 1, 4)
    set_block(U, top_right, 1, 4)
    # B-part: B22
    set_block(V, bottom_right, 1, 4)
    # C-part: contributes to C11 (-1) and C12 (+1)
    set_block(W, top_left, -1, 4)  # C11
    set_block(W, top_right, 1, 4)  # C12

    # M6 = (A21 - A11)(B11 + B12)
    # A-part: A21 - A11
    set_block(U, bottom_left, 1, 5)
    set_block(U, top_left, -1, 5)
    # B-part: B11 + B12
    set_block(V, top_left, 1, 5)
    set_block(V, top_right, 1, 5)
    # C-part: contributes to C22 (+1)
    set_block(W, bottom_right, 1, 5)  # C22

    # M7 = (A12 - A22)(B21 + B22)
    # A-part: A12 - A22
    set_block(U, top_right, 1, 6)
    set_block(U, bottom_right, -1, 6)
    # B-part: B21 + B22
    set_block(V, bottom_left, 1, 6)
    set_block(V, bottom_right, 1, 6)
    # C-part: contributes to C11 (+1)
    set_block(W, top_left, 1, 6)  # C11

    return U, V, W


def _schoolbook_nm_p(n, m, p):
    """
    Returns (U, V, W) for n x m x p matrix multiplication using schoolbook method.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5

    # We decompose the 4x5x5 multiplication into blocks:
    # Split columns of A and rows of B into two parts: first 4, last 1
    # A (4x5) = [A0 (4x4), A1 (4x1)]
    # B (5x5) = [[B00 (4x4), B01 (4x1)], [B10 (1x4), B11 (1x1)]]
    # C (4x5) = [C0 (4x4), C1 (4x1)]
    #
    # C0 = A0 * B00 + A1 * B10
    # C1 = A0 * B01 + A1 * B11
    #
    # Term 1: A0 (4x4) * B00 (4x4) -> C0 (4x4) : Use Strassen, rank 7
    # Term 2: A1 (4x1) * B10 (1x4) -> C0 (4x4) : Schoolbook 4x1x4, rank 16
    # Term 3: A0 (4x4) * B01 (4x1) -> C1 (4x1) : Schoolbook 4x4x1, rank 16
    # Term 4: A1 (4x1) * B11 (1x1) -> C1 (4x1) : Schoolbook 4x1x1, rank 4
    #
    # Total rank = 7 + 16 + 16 + 4 = 43

    # Generate sub-decompositions
    U1, V1, W1 = _strassen_4x4x4()  # rank 7
    U2, V2, W2 = _schoolbook_nm_p(4, 1, 4)  # rank 16
    U3, V3, W3 = _schoolbook_nm_p(4, 4, 1)  # rank 16
    U4, V4, W4 = _schoolbook_nm_p(4, 1, 1)  # rank 4

    total_rank = 7 + 16 + 16 + 4

    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    col_offset = 0

    # Term 1: A0 (4x4) * B00 (4x4) -> C0 (4x4)
    # A0 indices: A[i, j] for i in 0..3, j in 0..3 -> global U index i*5 + j
    # B00 indices: B[j, k] for j in 0..3, k in 0..3 -> global V index j*5 + k
    # C0 indices: C[i, k] for i in 0..3, k in 0..3 -> global W index k*4 + i

    for c in range(7):
        for row in range(16):
            val = U1[row, c]
            if val != 0:
                # row = i*4 + j for A0
                i = row // 4
                j = row % 4
                U[i * 5 + j, col_offset + c] = val

        for row in range(16):
            val = V1[row, c]
            if val != 0:
                # row = j*4 + k for B00
                j = row // 4
                k = row % 4
                V[j * 5 + k, col_offset + c] = val

        for row in range(16):
            val = W1[row, c]
            if val != 0:
                # row = k*4 + i for C0
                k = row // 4
                i = row % 4
                W[k * 4 + i, col_offset + c] = val

    col_offset += 7

    # Term 2: A1 (4x1) * B10 (1x4) -> C0 (4x4)
    # A1 indices: A[i, 4] for i in 0..3 -> global U index i*5 + 4
    # B10 indices: B[4, k] for k in 0..3 -> global V index 4*5 + k = 20 + k
    # C0 indices: C[i, k] for i in 0..3, k in 0..3 -> global W index k*4 + i

    for c in range(16):
        # U2 has shape (4*1, 16) = (4, 16)
        # U2[i, c] corresponds to A[i, 4]
        for i in range(4):
            val = U2[i, c]
            if val != 0:
                U[i * 5 + 4, col_offset + c] = val

        # V2 has shape (1*4, 16) = (4, 16)
        # V2[k, c] corresponds to B[4, k]
        for k in range(4):
            val = V2[k, c]
            if val != 0:
                V[20 + k, col_offset + c] = val

        # W2 has shape (4*4, 16) = (16, 16)
        # W2[k*4 + i, c] corresponds to C[i, k]
        for row in range(16):
            val = W2[row, c]
            if val != 0:
                k = row // 4
                i = row % 4
                W[k * 4 + i, col_offset + c] = val

    col_offset += 16

    # Term 3: A0 (4x4) * B01 (4x1) -> C1 (4x1)
    # A0 indices: A[i, j] for i in 0..3, j in 0..3 -> global U index i*5 + j
    # B01 indices: B[j, 4] for j in 0..3 -> global V index j*5 + 4
    # C1 indices: C[i, 4] for i in 0..3 -> global W index 4*4 + i = 16 + i

    for c in range(16):
        # U3 has shape (4*4, 16) = (16, 16)
        # U3[i*4 + j, c] corresponds to A[i, j]
        for row in range(16):
            val = U3[row, c]
            if val != 0:
                i = row // 4
                j = row % 4
                U[i * 5 + j, col_offset + c] = val

        # V3 has shape (4*1, 16) = (4, 16)
        # V3[j, c] corresponds to B[j, 4]
        for j in range(4):
            val = V3[j, c]
            if val != 0:
                V[j * 5 + 4, col_offset + c] = val

        # W3 has shape (1*4, 16) = (4, 16)
        # W3[i, c] corresponds to C[i, 4]
        for i in range(4):
            val = W3[i, c]
            if val != 0:
                W[16 + i, col_offset + c] = val

    col_offset += 16

    # Term 4: A1 (4x1) * B11 (1x1) -> C1 (4x1)
    # A1 indices: A[i, 4] for i in 0..3 -> global U index i*5 + 4
    # B11 indices: B[4, 4] -> global V index 4*5 + 4 = 24
    # C1 indices: C[i, 4] for i in 0..3 -> global W index 16 + i

    for c in range(4):
        # U4 has shape (4*1, 4) = (4, 4)
        # U4[i, c] corresponds to A[i, 4]
        for i in range(4):
            val = U4[i, c]
            if val != 0:
                U[i * 5 + 4, col_offset + c] = val

        # V4 has shape (1*1, 4) = (1, 4)
        # V4[0, c] corresponds to B[4, 4]
        val = V4[0, c]
        if val != 0:
            V[24, col_offset + c] = val

        # W4 has shape (1*4, 4) = (4, 4)
        # W4[i, c] corresponds to C[i, 4]
        for i in range(4):
            val = W4[i, c]
            if val != 0:
                W[16 + i, col_offset + c] = val

    col_offset += 4

    # Convert to lists for the expected return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
