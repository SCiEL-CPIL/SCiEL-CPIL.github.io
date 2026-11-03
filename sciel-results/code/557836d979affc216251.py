import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook rank = n*m*p = 105
    # Use Strassen 2x2x2 blocks where possible
    # Strassen terms for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7

    # For each Strassen block covering rows [i0, i0+1] of A, cols [j0, j0+1] of A,
    # and rows [k0, k0+1] of B:
    # We need to express in terms of the global U, V, W matrices.
    # U[i*m+j] corresponds to A[i,j]
    # V[j*p+k] corresponds to B[j,k]
    # W[k*n+i] corresponds to C[k*n+i] i.e. C[i,k] in standard notation
    # Wait, the convention is W[k*n+i, r] for the output.
    # So W index = k*n + i where k is column of C (which is p-index) and i is row of C (which is n-index).

    # Actually let me re-read: "W[k*n+i,r]"
    # So W has shape (p*n, rank). Index k*n+i means: for output C[i,k], the W row is k*n+i.
    # U has shape (n*m, rank). U[i*m+j] corresponds to input A[i,j].
    # V has shape (m*p, rank). V[j*p+k] corresponds to input B[j,k].

    # The tensor equation is:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r U[i*m+j... wait no.
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    # Let me build the rank-99 solution.

    rank = 99
    U = [[0] * rank for _ in range(n * m)]  # 15 x 99
    V = [[0] * rank for _ in range(m * p)]  # 35 x 99
    W = [[0] * rank for _ in range(p * n)]  # 21 x 99

    col = 0

    # Define Strassen block placement
    # i-blocks: [0,1] (one block, since n=3, rows 0-1 form a 2x2 block, row 2 is leftover)
    # j-blocks: [0,1], [2,3] (two blocks, since m=5, cols 0-1 and 2-3, col 4 is leftover)
    # k-blocks: [0,1], [2,3], [4,5] (three blocks, since p=7, rows 0-1, 2-3, 4-5, row 6 is leftover)

    i_blocks = [(0, 1)]
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]

    # For each combination of i_block, j_block, k_block, add 7 Strassen terms
    for i0, i1 in i_blocks:
        for j0, j1 in j_blocks:
            for k0, k1 in k_blocks:
                # A indices: (i0,j0), (i0,j1), (i1,j0), (i1,j1)
                # B indices: (j0,k0), (j0,k1), (j1,k0), (j1,k1)
                # C indices: (i0,k0), (i0,k1), (i1,k0), (i1,k1)

                # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
                U[i0 * m + j0][col] += 1
                U[i1 * m + j1][col] += 1
                V[j0 * p + k0][col] += 1
                V[j1 * p + k1][col] += 1
                W[k0 * n + i0][col] += 1   # C[i0,k0]
                W[k1 * n + i0][col] -= 1   # C[i0,k1]
                W[k0 * n + i1][col] -= 1   # C[i1,k0]
                W[k1 * n + i1][col] += 1   # C[i1,k1]
                col += 1

                # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
                U[i1 * m + j0][col] += 1
                U[i1 * m + j1][col] += 1
                V[j0 * p + k0][col] += 1
                W[k0 * n + i1][col] += 1   # C[i1,k0]
                W[k1 * n + i1][col] -= 1   # C[i1,k1]
                col += 1

                # M3 = A[i0,j0] * (B[j1,k0]-B[j1,k1])
                U[i0 * m + j0][col] += 1
                V[j1 * p + k0][col] += 1
                V[j1 * p + k1][col] -= 1
                W[k0 * n + i0][col] += 1   # C[i0,k0]
                W[k1 * n + i0][col] -= 1   # C[i0,k1]
                col += 1

                # M4 = A[i1,j1] * (B[j0,k1]-B[j0,k0])
                U[i1 * m + j1][col] += 1
                V[j0 * p + k1][col] += 1
                V[j0 * p + k0][col] -= 1
                W[k0 * n + i0][col] += 1   # C[i0,k0]
                W[k1 * n + i1][col] += 1   # C[i1,k1]
                col += 1

                # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
                U[i0 * m + j0][col] += 1
                U[i0 * m + j1][col] += 1
                V[j1 * p + k1][col] += 1
                W[k0 * n + i0][col] -= 1   # C[i0,k0]
                W[k1 * n + i0][col] += 1   # C[i0,k1]
                col += 1

                # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
                U[i1 * m + j0][col] += 1
                U[i0 * m + j0][col] -= 1
                V[j0 * p + k0][col] += 1
                V[j0 * p + k1][col] += 1
                W[k0 * n + i1][col] += 1   # C[i1,k0]
                col += 1

                # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k0])
                U[i0 * m + j1][col] += 1
                U[i1 * m + j1][col] -= 1
                V[j1 * p + k1][col] += 1
                V[j1 * p + k0][col] += 1
                W[k1 * n + i0][col] += 1   # C[i0,k1]
                col += 1

    # Now add schoolbook terms for all (i,j,k) not covered by any Strassen block
    # Covered i: {0,1}, j: {0,1,2,3}, k: {0,1,2,3,4,5}
    # Uncovered: i=2 (any j,k), or i in {0,1} and j=4 (any k), or i in {0,1} and j in {0,1,2,3} and k=6

    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this (i,j,k) is covered by a Strassen block
                in_i_block = (i in (0, 1))
                in_j_block = (j in (0, 1, 2, 3))
                in_k_block = (k in (0, 1, 2, 3, 4, 5))

                # A (i,j,k) term is covered if i is in an i_block, j is in a j_block,
                # AND k is in a k_block. More precisely, we need i to be in one of the
                # i_blocks, j in one of the j_blocks, k in one of the k_blocks.
                # i_blocks cover i in {0,1}, j_blocks cover j in {0,1,2,3}, k_blocks cover k in {0,1,2,3,4,5}
                if in_i_block and in_j_block and in_k_block:
                    continue  # Already handled by Strassen

                # Schoolbook term
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
