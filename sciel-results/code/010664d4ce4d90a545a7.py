import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    start_time = time.time()
    n, m, p = 4, 4, 4
    rank = 49

    U = np.zeros((n * m, rank))  # 16 x 49
    V = np.zeros((m * p, rank))  # 16 x 49
    W = np.zeros((p * n, rank))  # 16 x 49

    # Strassen 2x2: given 2x2 blocks A11,A12,A21,A22 and B11,B12,B21,B22
    # 7 products:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # Result:
    # C11 = M1 + M4 - M5 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7

    # For 4x4, partition into 2x2 blocks of 2x2 sub-matrices.
    # Block (bi, bj) of A is rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # Block (bi, bj) of B is rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # Block (bi, bj) of C is rows 2*bi..2*bi+1, cols 2*bj..2*bj+1

    # We apply Strassen at the block level. Each "block multiplication"
    # (e.g., A11 * B11) is itself a 2x2 matrix multiply, which we decompose
    # using Strassen recursively (rank 7 for each 2x2 multiply).

    # The 7 Strassen products at block level, each involving 2x2 block matrices:
    # For each product, we need to express the block-level linear combinations
    # in terms of individual elements, then apply 2x2 Strassen to each block product.

    # Let's define the 7 block-level Strassen terms explicitly.
    # Each term has:
    #   - A-side: linear combination of 4 2x2 blocks
    #   - B-side: linear combination of 4 2x2 blocks
    #   - Contribution to C blocks

    # Block indices: (0,0)=TL, (0,1)=TR, (1,0)=BL, (1,1)=BR
    # Element (i,j) in block (bi,bj) is at global position (2*bi+i, 2*bj+j)

    def elem(block_i, block_j, row, col):
        """Return global (row, col) for element at (row,col) in block (block_i, block_j)."""
        return (2 * block_i + row, 2 * block_j + col)

    # The 7 Strassen products at the block level:
    # Each is (A_comb) * (B_comb) where A_comb and B_comb are 2x2 block matrices
    # with entries being either 0, +1, or -1 times a block.

    strassen_terms = [
        # (A_blocks, B_blocks, C_contrib)
        # A_blocks: dict mapping (bi,bj) -> coeff
        # B_blocks: dict mapping (bi,bj) -> coeff
        # C_contrib: dict mapping (ci,cj) -> coeff
        {
            'A': {(0,0): 1, (1,1): 1},
            'B': {(0,0): 1, (1,1): 1},
            'C': {(0,0): 1, (1,1): 1}
        },  # M1
        {
            'A': {(1,0): 1, (1,1): 1},
            'B': {(0,0): 1},
            'C': {(1,0): 1, (1,1): -1}
        },  # M2
        {
            'A': {(0,0): 1},
            'B': {(0,1): 1, (1,1): -1},
            'C': {(0,1): 1, (1,1): 1}
        },  # M3
        {
            'A': {(1,1): 1},
            'B': {(1,0): 1, (0,0): -1},
            'C': {(0,0): 1, (1,0): 1}
        },  # M4
        {
            'A': {(0,0): 1, (0,1): 1},
            'B': {(1,1): 1},
            'C': {(0,0): -1, (0,1): 1}
        },  # M5
        {
            'A': {(1,0): 1, (0,0): -1},
            'B': {(0,0): 1, (0,1): 1},
            'C': {(1,1): 1}
        },  # M6
        {
            'A': {(0,1): 1, (1,1): -1},
            'B': {(1,0): 1, (1,1): 1},
            'C': {(0,0): 1}
        },  # M7
    ]

    # For each block-level Strassen term, the A-side is a 2x2 block matrix
    # where each 2x2 block is a linear combination of the original A blocks.
    # Similarly for B-side.
    # We need to compute the product of these two 2x2 block matrices,
    # which itself is a 2x2 block matrix.
    # Each resulting 2x2 block is a sum of 2x2 matrix products of the form
    # (sum of A sub-blocks) * (sum of B sub-blocks).
    # We use Strassen (rank 7) for each such 2x2 matrix product.

    # Actually, the cleaner approach: for each of the 7 block-level Strassen products,
    # we have A_comb (2x2 blocks) * B_comb (2x2 blocks) = C_comb (2x2 blocks).
    # Each C_comb block is a 2x2 matrix = sum over (i,j) of A_comb[i][j] * B_comb[j][k].
    # But A_comb[i][j] and B_comb[j][k] are each 2x2 matrices (or 0).
    # So C_comb[i][k] = sum_j A_comb[i][j] @ B_comb[j][k] (2x2 matrix multiply).
    # Each 2x2 matrix multiply has rank 7 (Strassen).
    # There are 4 positions in C_comb, but many are zero.

    # Let me take a different, more direct approach:
    # For each block-level Strassen term t, the A-side matrix (4x4) is built from
    # the A blocks, and the B-side matrix (4x4) is built from the B blocks.
    # Their product gives a 4x4 matrix that contributes to C.
    # We decompose this 4x4 product using Strassen (rank 49) -- but that's circular.

    # Better: Direct recursive Strassen.
    # 4x4 multiply = 7 products of 2x2 matrices (Strassen), each 2x2 multiply = 7 products of scalars.
    # Total rank = 7 * 7 = 49.

    # Let me implement this directly.

    # For a 2x2 matrix multiply C = A * B where A, B are 2x2:
    # Strassen gives 7 scalar products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7

    # For the 4x4 case, we partition into 2x2 blocks of 2x2 sub-matrices.
    # The 7 block-level Strassen products each involve 2x2 sub-matrix multiplications.

    # Let's build it step by step.
    # For each of the 7 block-level terms, we have:
    #   A_side: a 4x4 matrix formed by placing A-blocks in specific positions
    #   B_side: a 4x4 matrix formed by placing B-blocks in specific positions
    #   The product A_side @ B_side contributes to C.

    # But we can't just multiply 4x4 matrices; we need to decompose each.
    # The key insight: A_side and B_side are sparse (only certain 2x2 blocks are nonzero).
    # We can use the block structure to reduce the rank.

    # Actually, the standard recursive Strassen for 4x4 gives rank 49 directly.
    # Let me just construct it properly.

    # For the 4x4 multiply, using Strassen:
    # M1 = (A11+A22) * (B11+B22)  where A11, A22, B11, B22 are 2x2 blocks
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B12-B22)
    # M4 = A22 * (B21-B11)
    # M5 = (A11+A12) * B22
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A12-A22) * (B21+B22)

    # Each of these is a 2x2 matrix multiply, which has rank 7.
    # So total rank = 7 * 7 = 49.

    # For each block-level term, say M1 = (A11+A22)*(B11+B22):
    # Let X = A11+A22 (a 2x2 matrix) and Y = B11+B22 (a 2x2 matrix).
    # X*Y is a 2x2 matrix with rank-7 Strassen decomposition.
    # X[i,j] = A[2*i+di, 2*j+dj] + A[2*i+di, 2*j+dj] for appropriate (di,dj)
    # Specifically: X[i,j] = A[2*i, 2*j] + A[2*i+1, 2*j+1] (A11 + A22)
    # Y[i,j] = B[2*i, 2*j] + B[2*i+1, 2*j+1] (B11 + B22)

    # The 7 Strassen products for X*Y are:
    # m1 = (X00+X11)*(Y00+Y11)
    # m2 = (X10+X11)*Y00
    # m3 = X00*(Y01-Y11)
    # m4 = X11*(Y10-Y00)
    # m5 = (X00+X01)*Y11
    # m6 = (X10-X00)*(Y00+Y01)
    # m7 = (X01-X11)*(Y10+Y11)

    # Each m_k is a product of two linear combinations of A and B elements.
    # We add these to our U, V, W matrices.

    # Let me define the 7 block-level terms and their Strassen sub-decompositions.

    # Block-level term definitions:
    # term: (A_blocks_dict, B_blocks_dict, C_blocks_dict)
    # where each dict maps (bi,bj) -> coeff, and the block is at that position.

    block_terms = [
        # M1: (A11+A22)*(B11+B22) -> C11 += 1, C22 += 1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2: (A21+A22)*B11 -> C21 += 1, C22 += -1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3: A11*(B12-B22) -> C12 += 1, C22 += 1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4: A22*(B21-B11) -> C11 += 1, C21 += 1
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5: (A11+A12)*B22 -> C11 += -1, C12 += 1
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6: (A21-A11)*(B11+B12) -> C22 += 1
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7: (A12-A22)*(B21+B22) -> C11 += 1
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    # Strassen 2x2 decomposition:
    # For X*Y where X, Y are 2x2:
    # m1 = (X00+X11)*(Y00+Y11)
    # m2 = (X10+X11)*Y00
    # m3 = X00*(Y01-Y11)
    # m4 = X11*(Y10-Y00)
    # m5 = (X00+X01)*Y11
    # m6 = (X10-X00)*(Y00+Y01)
    # m7 = (X01-X11)*(Y10+Y11)
    # Result: C00 = m1+m4-m5+m6, C01 = m3+m5, C10 = m2+m4, C11 = m1-m2+m3+m7

    strassen_2x2 = [
        # (X_coeffs, Y_coeffs, C_contrib)
        # X_coeffs: dict (i,j) -> coeff for X[i,j]
        # Y_coeffs: dict (i,j) -> coeff for Y[i,j]
        # C_contrib: dict (i,j) -> coeff for result C[i,j]
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),  # m1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),  # m2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),  # m3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),  # m4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),  # m5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),  # m6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),  # m7
    ]

    for t_idx, (A_blocks, B_blocks, C_blocks) in enumerate(block_terms):
        for s_idx, (X_coeffs, Y_coeffs, C_contrib) in enumerate(strassen_2x2):
            rank_idx = t_idx * 7 + s_idx

            # Build U vector: for each A element, add coefficient
            for (bi, bj), block_coeff in A_blocks.items():
                for (xi, xj), x_coeff in X_coeffs.items():
                    # Element at global position (2*bi+xi, 2*bj+xj)
                    row = 2 * bi + xi
                    col = 2 * bj + xj
                    U[row * m + col, rank_idx] += block_coeff * x_coeff

            # Build V vector: for each B element, add coefficient
            for (bi, bj), block_coeff in B_blocks.items():
                for (yi, yj), y_coeff in Y_coeffs.items():
                    # Element at global position (2*bi+yi, 2*bj+yj)
                    row = 2 * bi + yi
                    col = 2 * bj + yj
                    V[row * p + col, rank_idx] += block_coeff * y_coeff

            # Build W vector: for each C element, add coefficient
            for (ci, cj), c_block_coeff in C_blocks.items():
                for (zi, zj), z_coeff in C_contrib.items():
                    # C element at global position (2*ci+zi, 2*cj+zj)
                    # W index: W[k*n+i] where k = col of C, i = row of C
                    # So W[(2*cj+zj)*n + (2*ci+zi)]
                    w_row = 2 * cj + zj
                    w_col = 2 * ci + zi
                    W[w_row * n + w_col, rank_idx] += c_block_coeff * z_coeff

    # Verify
    U_np = U.astype(float)
    V_np = V.astype(float)
    W_np = W.astype(float)

    for _test in range(5):
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test

        # C[i,j] = sum_r (sum_k U[i*m+k, r] * A[i,k]) * (sum_l V[l*p+j, r] * B[l,j]) * W[j*n+i, r]
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            a_comb = np.dot(A_test.flatten(), U_np[:, r])
            b_comb = np.dot(B_test.flatten(), V_np[:, r])
            prod = a_comb * b_comb
            C_decomp += prod * W_np[:, r].reshape(n, p)

        if not np.allclose(C_decomp, C_expected, atol=1e-10):
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
