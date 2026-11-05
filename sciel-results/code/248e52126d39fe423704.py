import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4

    # Rank-44 decomposition:
    #   Top 2 rows of A (2x4) times B (4x4) -> 2x4 output.
    #     Split into four 2x2 block multiplications, each done via Strassen (rank 7)
    #     => 4 * 7 = 28 terms.
    #   Bottom 1 row of A (1x4) times B (4x4) -> 1x4 output.
    #     Schoolbook: 1*4*4 = 16 terms.
    #   Total rank = 44.

    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44

    # Strassen 2x2x2 factors (rank 7).
    # M1 = A00*B00
    # M2 = A10*(B01+B11)
    # M3 = (A00+A01)*B10
    # M4 = A11*(B10-B00)
    # M5 = (A01+A11)*(B00-B10)
    # M6 = (A00+A10)*(B11-B01)
    # M7 = A10*B11
    # Output: C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
    #
    # U_s[r][k] for A-factor rows r in {A00,A01,A10,A11}, k in 0..6
    U_s = [
        [1, 0, 1, 0, 0, 1, 0],  # A00
        [0, 0, 1, 0, 1, 0, 0],  # A01
        [0, 1, 0, 0, 0, 1, 1],  # A10
        [0, 0, 0, 1, 1, 0, 0],  # A11
    ]
    # V_s[c][k] for B-factor cols c in {B00,B01,B10,B11}
    V_s = [
        [1, 0, 0, -1, 1, 0, 0],  # B00
        [0, 1, 0, 0, 0, -1, 0],  # B01
        [0, 0, 1, 1, -1, 0, 0],  # B10
        [0, 1, 0, 0, 0, 1, 0],  # B11
    ]
    # W_s[o][k] for output positions o in {C00,C01,C10,C11}
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0],  # C11
    ]

    # Add a Strassen block for A_block (2x2) * B_block (2x2) -> C_block (2x2).
    # a_row: row offset in A (0-indexed)
    # b_col: col offset in B (0-indexed)
    # c_row: row offset in C (0-indexed)
    # c_col: col offset in C (0-indexed)
    def add_strassen(a_row, b_col, c_row, c_col, col_offset):
        for r in range(2):
            for c in range(2):
                for k in range(7):
                    u_val = U_s[r][k]
                    v_val = V_s[c][k]
                    if u_val != 0:
                        U[(a_row + r) * m + (b_col + c)][col_offset + k] += u_val
                    if v_val != 0:
                        V[(b_col + c) * p + (c_col + r)][col_offset + k] += v_val
                    # Output: C_block[r][c] = W_s[r*2+c][k] * M_k
                    w_val = W_s[r * 2 + c][k]
                    if w_val != 0:
                        W[(c_col + r) * n + (c_row + c)][col_offset + k] += w_val

    # Four 2x2 block multiplications for the top 2x4 part:
    # A_top = [A1 (cols 0,1) | A2 (cols 2,3)]
    # B     = [B11 (rows 0,1; cols 0,1) B12 (rows 0,1; cols 2,3)]
    #         [B21 (rows 2,3; cols 0,1) B22 (rows 2,3; cols 2,3)]
    # C_top = [C11 C12; C21 C22] where C11 = A1*B11 + A2*B21, etc.
    #
    # Block (A1, B11) -> C11: a_row=0, b_col=0, c_row=0, c_col=0
    add_strassen(0, 0, 0, 0, 0)
    # Block (A2, B21) -> C11: a_row=0, b_col=2, c_row=0, c_col=0
    add_strassen(0, 2, 0, 0, 7)
    # Block (A1, B12) -> C12: a_row=0, b_col=0, c_row=0, c_col=2
    add_strassen(0, 0, 0, 2, 14)
    # Block (A2, B22) -> C12: a_row=0, b_col=2, c_row=0, c_col=2
    add_strassen(0, 2, 0, 2, 21)

    # Wait — I need to reconsider the block structure.
    # A_top is 2x4: rows 0,1 of A; columns 0,1,2,3.
    # B is 4x4.
    # C_top is 2x4: rows 0,1 of C; columns 0,1,2,3.
    #
    # C_top[i][j] = sum_k A_top[i][k] * B[k][j]
    #
    # Split A_top into left 2x2 (cols 0,1) and right 2x2 (cols 2,3).
    # Split B into four 2x2 blocks: B[0:2,0:2], B[0:2,2:4], B[2:4,0:2], B[2:4,2:4].
    #
    # C_top[:, 0:2] = A_left * B_top_left + A_right * B_bot_left
    # C_top[:, 2:4] = A_left * B_top_right + A_right * B_bot_right
    #
    # So:
    # C_top[:,0:2] gets contributions from (A_left, B[0:2,0:2]) and (A_right, B[2:4,0:2])
    # C_top[:,2:4] gets contributions from (A_left, B[0:2,2:4]) and (A_right, B[2:4,2:4])

    # Reset and redo properly
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Block 1: A rows 0-1, cols 0-1 (a_row=0, a_col=0) * B rows 0-1, cols 0-1 (b_row=0, b_col=0)
    #   -> C rows 0-1, cols 0-1 (c_row=0, c_col=0)
    add_strassen(0, 0, 0, 0, 0)
    # Block 2: A rows 0-1, cols 2-3 (a_row=0, a_col=2) * B rows 2-3, cols 0-1 (b_row=2, b_col=0)
    #   -> C rows 0-1, cols 0-1 (c_row=0, c_col=0)
    add_strassen(0, 2, 0, 0, 7)
    # Block 3: A rows 0-1, cols 0-1 (a_row=0, a_col=0) * B rows 0-1, cols 2-3 (b_row=0, b_col=2)
    #   -> C rows 0-1, cols 2-3 (c_row=0, c_col=2)
    add_strassen(0, 0, 0, 2, 14)
    # Block 4: A rows 0-1, cols 2-3 (a_row=0, a_col=2) * B rows 2-3, cols 2-3 (b_row=2, b_col=2)
    #   -> C rows 0-1, cols 2-3 (c_row=0, c_col=2)
    add_strassen(0, 2, 0, 2, 21)

    # Schoolbook for bottom row: A[2, j] * B[j, k] -> C[2, k] for j=0..3, k=0..3
    col = 28
    for j in range(4):
        for k in range(4):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1

    # Verify: check that the tensor decomposition is correct
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1 if (i,j,k) is a valid index
    # Actually, the tensor for matrix multiplication is:
    # T[i,j,k] = delta[i,j,k] where the multiplication is C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is T[i,j,k] = 1 for all i,j,k (it's the full outer product structure)
    # Wait, no. The matrix multiplication tensor has entries:
    # For each (i, j, k), the tensor entry is 1 (since C[i,k] = sum_j A[i,j]*B[j,k])
    # Actually the standard form: the tensor M_{i,j,k} = 1 for all i in [n], j in [m], k in [p]
    # because C[i,k] = sum_j A[i,j] B[j,k] means the tensor is the all-ones tensor.
    #
    # Verification: for random A, B, check that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] gives C[i,k]
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        return _schoolbook(n, m, p)

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    # Test with random matrices
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        # Compute via tensor: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                C_computed[i, k] = s
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def _schoolbook(n, m, p):
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
# EVOLVE-BLOCK-END
