"""Program for Matrix multiplication 4,4,5.

Uses a recursive Strassen-style decomposition for the 4x4x4 core
(rank 49) plus 4 extra scalar terms for the 5th column of B,
giving total rank 53.
"""
import numpy as np

DIMENSIONS = [4, 4, 5]


def _strassen_2x2():
    """Return U, V, W for 2x2x2 matrix multiplication (Strassen).
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    n = 2
    rank = 7
    U = np.zeros((n * n, rank))
    V = np.zeros((n * n, rank))
    W = np.zeros((n * n, rank))

    # Strassen formulas for 2x2:
    # P1 = (a00+a11)*(b00+b11)
    # P2 = (a10+a11)*b00
    # P3 = a00*(b01-b11)
    # P4 = (a00+a01)*b11
    # P5 = (a10-a00)*(b00+b01)
    # P6 = (a01-a11)*(b10+b11)
    # P7 = (a00+a11-a10-a01)*(b10+b11-b00-b01)

    # Result:
    # c00 = P1 + P4 - P5 - P7
    # c01 = P3 + P5
    # c10 = P2 + P4
    # c11 = P1 - P2 + P3 + P6

    # U factors (A side):
    # U index: i*2+j
    # P1: (a00+a11) -> U[0,0]=1, U[3,0]=1
    # P2: (a10+a11) -> U[2,1]=1, U[3,1]=1
    # P3: a00 -> U[0,2]=1
    # P4: (a00+a01) -> U[0,3]=1, U[1,3]=1
    # P5: (a10-a00) -> U[2,4]=1, U[0,4]=-1
    # P6: (a01-a11) -> U[1,5]=1, U[3,5]=-1
    # P7: (a00+a11-a10-a01) -> U[0,6]=1, U[3,6]=1, U[2,6]=-1, U[1,6]=-1

    U[0, 0] = 1; U[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    U[0, 2] = 1
    U[0, 3] = 1; U[1, 3] = 1
    U[2, 4] = 1; U[0, 4] = -1
    U[1, 5] = 1; U[3, 5] = -1
    U[0, 6] = 1; U[3, 6] = 1; U[2, 6] = -1; U[1, 6] = -1

    # V factors (B side):
    # V index: j*2+k
    # P1: (b00+b11) -> V[0,0]=1, V[3,0]=1
    # P2: b00 -> V[0,1]=1
    # P3: (b01-b11) -> V[1,2]=1, V[3,2]=-1
    # P4: b11 -> V[3,3]=1
    # P5: (b00+b01) -> V[0,4]=1, V[1,4]=1
    # P6: (b10+b11) -> V[2,5]=1, V[3,5]=1
    # P7: (b10+b11-b00-b01) -> V[2,6]=1, V[3,6]=1, V[0,6]=-1, V[1,6]=-1

    V[0, 0] = 1; V[3, 0] = 1
    V[0, 1] = 1
    V[1, 2] = 1; V[3, 2] = -1
    V[3, 3] = 1
    V[0, 4] = 1; V[1, 4] = 1
    V[2, 5] = 1; V[3, 5] = 1
    V[2, 6] = 1; V[3, 6] = 1; V[0, 6] = -1; V[1, 6] = -1

    # W factors (C side):
    # W index: k*2+i
    # P1: c00+=1, c11+=1 -> W[0,0]=1, W[3,0]=1
    # P2: c10+=1, c11+=-1 -> W[2,1]=1, W[3,1]=-1
    # P3: c01+=1, c11+=1 -> W[1,2]=1, W[3,2]=1
    # P4: c00+=1, c10+=1 -> W[0,3]=1, W[2,3]=1
    # P5: c00+=-1, c01+=1 -> W[0,4]=-1, W[1,4]=1
    # P6: c11+=1 -> W[3,5]=1
    # P7: c00+=-1 -> W[0,6]=-1

    W[0, 0] = 1; W[3, 0] = 1
    W[2, 1] = 1; W[3, 1] = -1
    W[1, 2] = 1; W[3, 2] = 1
    W[0, 3] = 1; W[2, 3] = 1
    W[0, 4] = -1; W[1, 4] = 1
    W[3, 5] = 1
    W[0, 6] = -1

    return U, V, W


def _expand_strassen(U_small, V_small, W_small, block_size):
    """Expand a Strassen decomposition from block_size x block_size 
    to 2*block_size x 2*block_size.
    
    U_small: (block_size^2, 7)
    V_small: (block_size^2, 7)
    W_small: (block_size^2, 7)
    
    Returns U, V, W of shape ((2*block_size)^2, 49).
    """
    n = 2 * block_size
    rank = 49
    U = np.zeros((n * n, rank))
    V = np.zeros((n * n, rank))
    W = np.zeros((n * n, rank))

    # Block indices for A (n x n):
    # Block (0,0): rows 0..block_size-1, cols 0..block_size-1
    # Block (0,1): rows 0..block_size-1, cols block_size..2*block_size-1
    # Block (1,0): rows block_size..2*block_size-1, cols 0..block_size-1
    # Block (1,1): rows block_size..2*block_size-1, cols block_size..2*block_size-1
    #
    # For each of the 7 Strassen terms, we need linear combinations of blocks.
    # U factor: A side. Each column of U_small corresponds to a scalar in A.
    # The block combination for A:
    # P1: A00 + A11
    # P2: A10 + A11
    # P3: A00
    # P4: A00 + A01
    # P5: A10 - A00
    # P6: A01 - A11
    # P7: A00 + A11 - A10 - A01
    #
    # U index: i*n + j. For block (bi, bj), i is in [bi*bs, (bi+1)*bs), j in [bj*bs, (bj+1)*bs).
    # U index = i*n + j = (bi*bs + ii)*n + (bj*bs + jj) where ii, jj in [0, bs).
    #
    # For each Strassen term t (0..6) and each base column c (0..6):
    # The new column index is t*7 + c.
    # We need to fill U, V, W for this column.
    
    bs = block_size
    # Block offsets
    # A blocks: (bi, bj) -> row offset bi*bs, col offset bj*bs
    # B blocks: (bj, bk) -> row offset bj*bs, col offset bk*bs
    # C blocks: (bi, bk) -> row offset bi*bs, col offset bk*bs

    # Coefficients for A blocks in each Strassen term
    # term t: coefficients for blocks (0,0), (0,1), (1,0), (1,1)
    A_coeffs = [
        [1, 0, 0, 1],   # P1: A00 + A11
        [0, 0, 1, 1],   # P2: A10 + A11
        [1, 0, 0, 0],   # P3: A00
        [1, 1, 0, 0],   # P4: A00 + A01
        [-1, 0, 1, 0],  # P5: A10 - A00
        [0, 1, 0, -1],  # P6: A01 - A11
        [1, -1, -1, 1], # P7: A00 - A01 - A10 + A11
    ]

    # Coefficients for B blocks in each Strassen term
    # term t: coefficients for blocks (0,0), (0,1), (1,0), (1,1)
    B_coeffs = [
        [1, 0, 0, 1],   # P1: B00 + B11
        [1, 0, 0, 0],   # P2: B00
        [0, 1, 0, -1],  # P3: B01 - B11
        [0, 0, 0, 1],   # P4: B11
        [1, 1, 0, 0],   # P5: B00 + B01
        [0, 0, 1, 1],   # P6: B10 + B11
        [-1, -1, 1, 1], # P7: B10 + B11 - B00 - B01
    ]

    # Coefficients for C blocks in each Strassen term
    # term t: coefficients for blocks (0,0), (0,1), (1,0), (1,1)
    C_coeffs = [
        [1, 0, 0, 1],   # P1: C00 + C11
        [0, 0, 1, -1],  # P2: C10 - C11
        [0, 1, 0, 1],   # P3: C01 + C11
        [1, 0, 1, 0],   # P4: C00 + C10
        [-1, 1, 0, 0],  # P5: -C00 + C01
        [0, 0, 0, 1],   # P6: C11
        [-1, 0, 0, 0],  # P7: -C00
    ]

    for t in range(7):
        for c in range(7):
            col = t * 7 + c
            # For each base entry in U_small (size bs*bs), scale by A_coeffs
            for ii in range(bs):
                for jj in range(bs):
                    base_idx = ii * bs + jj
                    val = U_small[base_idx, c]
                    if val != 0:
                        for bi in range(2):
                            for bj in range(2):
                                coeff = A_coeffs[t][bi * 2 + bj]
                                if coeff != 0:
                                    row = (bi * bs + ii) * n + (bj * bs + jj)
                                    U[row, col] += coeff * val

            for ii in range(bs):
                for jj in range(bs):
                    base_idx = ii * bs + jj
                    val = V_small[base_idx, c]
                    if val != 0:
                        for bi in range(2):
                            for bj in range(2):
                                coeff = B_coeffs[t][bi * 2 + bj]
                                if coeff != 0:
                                    row = (bi * bs + ii) * n + (bj * bs + jj)
                                    V[row, col] += coeff * val

            for ii in range(bs):
                for jj in range(bs):
                    base_idx = ii * bs + jj
                    val = W_small[base_idx, c]
                    if val != 0:
                        for bi in range(2):
                            for bj in range(2):
                                coeff = C_coeffs[t][bi * 2 + bj]
                                if coeff != 0:
                                    row = (bi * bs + ii) * n + (bj * bs + jj)
                                    W[row, col] += coeff * val

    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5

    # Build 4x4x4 Strassen decomposition
    U2, V2, W2 = _strassen_2x2()
    U4, V4, W4 = _expand_strassen(U2, V2, W2, 2)

    # Now handle the 5th column of B (index 4)
    # We need C[:, 4] = A @ B[:, 4]
    # This is a rank-4 decomposition:
    # For each j in 0..3:
    #   U[i*m+j, 49+j] = 1 if we index A row i and column j
    #   V[j*p+4, 49+j] = 1
    #   W[4*n+i, 49+j] = 1
    # But wait, the W index convention is W[k*n+i, c] where k is column of C (0..p-1)
    # and i is row of C (0..n-1). For C[:, 4], k=4.
    # So W[4*n+i, 49+j] = 1.
    
    total_rank = 49 + 4
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    # Copy the 4x4x4 part
    U[:, :49] = U4
    V[:, :49] = V4
    W[:, :49] = W4

    # Add the 4 terms for the 5th column
    for j in range(m):
        for i in range(n):
            U[i * m + j, 49 + j] = 1
        V[j * p + 4, 49 + j] = 1
        W[4 * n + i, 49 + j] = 1

    # Convert to lists of integers
    U_list = U.astype(int).tolist()
    V_list = V.astype(int).tolist()
    W_list = W.astype(int).tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
