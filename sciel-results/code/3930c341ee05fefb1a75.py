"""Rank-minimal exact tensor decomposition for 2x4 by 4x4 matrix multiplication."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    Uses Strassen's algorithm on 2x2 blocks to achieve rank 28 (vs 32 schoolbook).
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28  # 4 blocks * 7 Strassen multiplications
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28

    # Strassen coefficients for 2x2 blocks
    # For matrices A (2x2), B (2x2) partitioned into:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A*B = [[C00, C01], [C10, C11]]
    #
    # Strassen's 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # Then:
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 + M5

    def add_strassen(a_idx, b_idx, c_idx, col_offset):
        """
        Add Strassen decomposition for one 2x2 block multiplication.
        a_idx: [A00, A01, A10, A11] global indices
        b_idx: [B00, B01, B10, B11] global indices
        c_idx: [C00, C01, C10, C11] global indices
        col_offset: starting column in U, V, W
        """
        A00, A01, A10, A11 = a_idx
        B00, B01, B10, B11 = b_idx
        C00, C01, C10, C11 = c_idx

        # M1 = (A00+A11)*(B00+B11)
        col = col_offset + 0
        U[A00][col] += 1
        U[A11][col] += 1
        V[B00][col] += 1
        V[B11][col] += 1
        W[C00][col] += 1
        W[C11][col] += 1

        # M2 = (A10+A11)*B00
        col = col_offset + 1
        U[A10][col] += 1
        U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1
        W[C11][col] += 1

        # M3 = A00*(B01-B11)
        col = col_offset + 2
        U[A00][col] += 1
        V[B01][col] += 1
        V[B11][col] -= 1
        W[C01][col] += 1
        W[C11][col] -= 1

        # M4 = A11*(B10-B00)
        col = col_offset + 3
        U[A11][col] += 1
        V[B10][col] += 1
        V[B00][col] -= 1
        W[C00][col] -= 1
        W[C10][col] += 1

        # M5 = (A00+A01)*B11
        col = col_offset + 4
        U[A00][col] += 1
        U[A01][col] += 1
        V[B11][col] += 1
        W[C01][col] += 1
        W[C11][col] += 1

        # M6 = (A10-A00)*(B00+B01)
        col = col_offset + 5
        U[A10][col] += 1
        U[A00][col] -= 1
        V[B00][col] += 1
        V[B01][col] += 1
        W[C00][col] += 1

        # M7 = (A01-A11)*(B10+B11)
        col = col_offset + 6
        U[A01][col] += 1
        U[A11][col] -= 1
        V[B10][col] += 1
        V[B11][col] += 1
        W[C00][col] += 1
        W[C10][col] += 1

    # Block 1: A[:, 0:2] * B[0:2, :] -> C[:, 0:2]
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # C indices: C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3], 0)

    # Block 2: A[:, 2:4] * B[0:2, :] -> C[:, 2:4]
    # A indices: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # C indices: C[0,2]=4, C[0,3]=6, C[1,2]=5, C[1,3]=7
    add_strassen([2, 3, 6, 7], [0, 1, 4, 5], [4, 6, 5, 7], 7)

    # Block 3: A[:, 0:2] * B[2:4, :] -> C[:, 0:2]
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # B indices: B[2,0]=8, B[2,1]=9, B[3,0]=12, B[3,1]=13
    # C indices: C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    add_strassen([0, 1, 4, 5], [8, 9, 12, 13], [0, 2, 1, 3], 14)

    # Block 4: A[:, 2:4] * B[2:4, :] -> C[:, 2:4]
    # A indices: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    # B indices: B[2,0]=8, B[2,1]=9, B[3,0]=12, B[3,1]=13
    # C indices: C[0,2]=4, C[0,3]=6, C[1,2]=5, C[1,3]=7
    add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [4, 6, 5, 7], 21)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
