"""Strassen-based rank-32 decomposition for 2x4x5 matrix multiplication."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Build a rank-32 exact tensor decomposition for multiplying a 2x4 matrix A
    by a 4x5 matrix B.

    Strategy:
    - Split B into B1 (cols 0-3) and B2 (col 4).
    - For B1 (2x4 * 4x4): use Strassen's rank-7 decomposition on four 2x2x2
      blocks. This gives rank 28.
    - For B2 (2x4 * 4x1): use schoolbook rank-4 decomposition.
    Total rank = 28 + 4 = 32.

    Conventions:
    - U has shape (n*m, rank) = (8, 32), indexed by U[i*m+j, c]
    - V has shape (m*p, rank) = (20, 32), indexed by V[j*p+k, c]
    - W has shape (p*n, rank) = (10, 32), indexed by W[k*n+i, c]
    """
    n, m, p = 2, 4, 5
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen decomposition for 2x2x2 blocks
    # For block with A rows (i0, i1), A cols (j0, j1), B rows (j0, j1), B cols (k0, k1):
    # The 7 terms and their contributions:
    #
    # Term 1: (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
    #   U: i0,j0=1; i1,j1=1
    #   V: j0,k0=1; j1,k1=1
    #   W: k0,i0=1; k0,i1=1; k1,i0=1; k1,i1=1
    #
    # Term 2: (A[i1,j0]+A[i1,j1]) * B[j0,k0]
    #   U: i1,j0=1; i1,j1=1
    #   V: j0,k0=1
    #   W: k0,i0=1; k0,i1=1
    #
    # Term 3: A[i0,j0] * (B[j0,k1]-B[j1,k1])
    #   U: i0,j0=1
    #   V: j0,k1=1; j1,k1=-1
    #   W: k1,i0=1; k1,i1=1
    #
    # Term 4: A[i1,j1] * (B[j1,k0]-B[j0,k0])
    #   U: i1,j1=1
    #   V: j1,k0=1; j0,k0=-1
    #   W: k0,i0=1; k0,i1=1
    #
    # Term 5: (A[i0,j0]+A[i0,j1]) * B[j1,k1]
    #   U: i0,j0=1; i0,j1=1
    #   V: j1,k1=1
    #   W: k1,i0=1; k1,i1=1
    #
    # Term 6: (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
    #   U: i1,j0=1; i0,j0=-1
    #   V: j0,k0=1; j0,k1=1
    #   W: k0,i0=1; k0,i1=1; k1,i0=1; k1,i1=1
    #
    # Term 7: (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
    #   U: i0,j1=1; i1,j1=-1
    #   V: j1,k0=1; j1,k1=1
    #   W: k0,i0=1; k0,i1=1; k1,i0=1; k1,i1=1

    def add_term(col, u_entries, v_entries, w_entries):
        for (idx, val) in u_entries:
            U[idx][col] += val
        for (idx, val) in v_entries:
            V[idx][col] += val
        for (idx, val) in w_entries:
            W[idx][col] += val

    # Process each 2x2x2 block in the 4x4 part (k=0..3)
    # Block (i0,i1, j0,j1, k0,k1) where i0 in {0,1}, j0 in {0,2}, k0 in {0,2}
    block_col = 0
    for i0 in range(0, 2, 1):
        i1 = i0 + 1
        for j0 in range(0, 4, 2):
            j1 = j0 + 1
            for k0 in range(0, 4, 2):
                k1 = k0 + 1
                # Term 1
                add_term(block_col,
                    [(i0*m+j0, 1), (i1*m+j1, 1)],
                    [(j0*p+k0, 1), (j1*p+k1, 1)],
                    [(k0*n+i0, 1), (k0*n+i1, 1), (k1*n+i0, 1), (k1*n+i1, 1)])
                block_col += 1
                # Term 2
                add_term(block_col,
                    [(i1*m+j0, 1), (i1*m+j1, 1)],
                    [(j0*p+k0, 1)],
                    [(k0*n+i0, 1), (k0*n+i1, 1)])
                block_col += 1
                # Term 3
                add_term(block_col,
                    [(i0*m+j0, 1)],
                    [(j0*p+k1, 1), (j1*p+k1, -1)],
                    [(k1*n+i0, 1), (k1*n+i1, 1)])
                block_col += 1
                # Term 4
                add_term(block_col,
                    [(i1*m+j1, 1)],
                    [(j1*p+k0, 1), (j0*p+k0, -1)],
                    [(k0*n+i0, 1), (k0*n+i1, 1)])
                block_col += 1
                # Term 5
                add_term(block_col,
                    [(i0*m+j0, 1), (i0*m+j1, 1)],
                    [(j1*p+k1, 1)],
                    [(k1*n+i0, 1), (k1*n+i1, 1)])
                block_col += 1
                # Term 6
                add_term(block_col,
                    [(i1*m+j0, 1), (i0*m+j0, -1)],
                    [(j0*p+k0, 1), (j0*p+k1, 1)],
                    [(k0*n+i0, 1), (k0*n+i1, 1), (k1*n+i0, 1), (k1*n+i1, 1)])
                block_col += 1
                # Term 7
                add_term(block_col,
                    [(i0*m+j1, 1), (i1*m+j1, -1)],
                    [(j1*p+k0, 1), (j1*p+k1, 1)],
                    [(k0*n+i0, 1), (k0*n+i1, 1), (k1*n+i0, 1), (k1*n+i1, 1)])
                block_col += 1

    # Now block_col should be 28. Add the 4 terms for the k=4 column (schoolbook rank 4)
    # C[i,4] = sum_{j=0}^{3} A[i,j] * B[j,4]
    # We need 4 terms, one for each j.
    for j in range(4):
        # U: A[i,j] for i=0,1 -> rows i*m+j
        # V: B[j,4] -> row j*p+4
        # W: C[i,4] for i=0,1 -> rows 4*n+i = 8+i
        add_term(block_col,
            [(0*m+j, 1), (1*m+j, 1)],
            [(j*p+4, 1)],
            [(4*n+0, 1), (4*n+1, 1)])
        block_col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
