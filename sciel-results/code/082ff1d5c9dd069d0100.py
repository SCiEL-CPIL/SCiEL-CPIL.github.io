"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Use Strassen's algorithm recursively:
    - Split 6x6 into four 3x3 blocks
    - Multiply using Strassen's 7 multiplications of 3x3 matrices
    - Each 3x3 multiplication is done by the standard schoolbook method (rank 27)
    Total rank = 7 * 27 = 189
    """
    n, m, p = DIMENSIONS
    assert n == m == p == 6
    
    # Strassen's algorithm for 6x6 using 3x3 sub-matrices
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # where each sub-block is 3x3
    #
    # C11 = M1 - M4 + M6 + M7
    # C12 = M2 + M3
    # C21 = M5 + M4
    # C22 = M1 - M2 + M3 + M5
    #
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # Each M_i is a 3x3 matrix product, requiring 27 scalar multiplications
    # Total: 7 * 27 = 189 multiplications
    
    rank = 7 * 27  # 189
    
    # For each of the 7 Strassen products, we need a rank-27 schoolbook decomposition
    # of a 3x3 matrix multiplication.
    #
    # The full 6x6 multiplication tensor is:
    # T[i, j, k] = A[i,j] * B[j,k]
    # where i in [0,5], j in [0,5], k in [0,5]
    #
    # We partition the 6x6 matrices into 2x2 blocks of 3x3:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    #
    # For each Strassen product M_r = X_r * Y_r where X_r and Y_r are 3x3 matrices
    # formed from linear combinations of the 3x3 blocks, we use schoolbook.
    #
    # The U, V, W matrices have shape (36, 189), (36, 189), (36, 189)
    # U row index: i * 6 + j (i in [0,5], j in [0,5])
    # V row index: j * 6 + k (j in [0,5], k in [0,5])
    # W row index: k * 6 + i (k in [0,5], i in [0,5]) -- trace convention W[k*n+i, r]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen products as linear combinations
    # For each product r, we define:
    #   X_r[i, j] = sum of coefficients * A[block_i, block_j][i', j']
    #   Y_r[i, j] = sum of coefficients * B[block_i, block_j][i', j']
    #
    # Let's index the 3x3 blocks:
    # A11: rows 0-2, cols 0-2  -> (0,0)
    # A12: rows 0-2, cols 3-5  -> (0,1)
    # A21: rows 3-5, cols 0-2  -> (1,0)
    # A22: rows 3-5, cols 3-5  -> (1,1)
    # Same for B.
    
    # For each Strassen product, define the linear combination coefficients
    # for each 3x3 block in X_r and Y_r
    #
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # For each product r, x_coeffs[block][i][j] and y_coeffs[block][i][j]
    # where block is in [(0,0), (0,1), (1,0), (1,1)]
    
    strassen_products = [
        # M1
        (
            {  # X = A11 + A22
                (0, 0): {(a, b): (1 if (a == 0 and b == 0) or (a == 3 and b == 3) else 0) for a in range(6) for b in range(6)},
            },
            {  # Y = B11 + B22
                (0, 0): {(a, b): (1 if (a == 0 and b == 0) or (a == 3 and b == 3) else 0) for a in range(6) for b in range(6)},
            }
        ),
    ]
    
    # Let me use a cleaner representation.
    # For each 3x3 block position (bi, bj) in {0,1}x{0,1}, the block covers
    # rows bi*3 to bi*3+2 and cols bj*3 to bj*3+2.
    #
    # For a 3x3 matrix product C = X * Y, the schoolbook decomposition has rank 27.
    # For each (i', j', k') in [0,2]^3, we have:
    #   U[row_X(i', j'), col] = 1
    #   V[row_Y(j', k'), col] = 1
    #   W[row_C(k', i'), col] = 1
    #
    # where row_X(i', j') = bi*3 + i', bj*3 + j' for the block at (bi, bj)
    # etc.
    
    # Let me redefine more carefully.
    
    # For each Strassen product r = 0..6:
    #   X_r is a 3x3 matrix = sum over blocks (bi,bj) of coeff_x[r][bi][bj] * A[bi,bj]
    #   Y_r is a 3x3 matrix = sum over blocks (bi,bj) of coeff_y[r][bi][bj] * B[bi,bj]
    #   M_r = X_r * Y_r is a 3x3 matrix
    #
    # The contribution of M_r to the output C is:
    #   C[bi_c, bj_c] += coeff_c[r][bi_c][bj_c] * M_r
    #
    # For the schoolbook decomposition of M_r = X_r * Y_r:
    #   For each (i', j', k') in [0,2]^3:
    #     The scalar product X_r[i', j'] * Y_r[j', k'] contributes to M_r[k', i']
    #     X_r[i', j'] = sum_{bi,bj} coeff_x[r][bi][bj] * A[bi*3+i', bj*3+j']
    #     Y_r[j', k'] = sum_{bi,bj} coeff_y[r][bi][bj] * B[bi*3+j', bj*3+k']
    #
    # In the tensor decomposition:
    #   U[i, j, r'] where i = bi*3+i', j = bj*3+j', r' = r*27 + i'*27 + j'*9 + k'
    #     U value = coeff_x[r][bi][bj]
    #   V[j, k, r'] where j = bi*3+j', k = bj*3+k', r' = r*27 + i'*27 + j'*9 + k'
    #     V value = coeff_y[r][bi][bj]
    #   W[k, i, r'] where k = bi_c*3+k', i = bi_c*3+i', r' = r*27 + i'*27 + j'*9 + k'
    #     W value = coeff_c[r][bi_c][bj_c]
    
    # Define coefficients for each Strassen product
    # coeff_x[r] is a dict from (bi, bj) to coefficient
    # Same for coeff_y[r] and coeff_c[r]
    
    coeff_x = [
        {(0,0): 1, (1,1): 1},  # M1: A11 + A22
        {(1,0): 1, (1,1): 1},  # M2: A21 + A22
        {(0,0): 1},            # M3: A11
        {(1,1): 1},            # M4: A22
        {(0,0): 1, (0,1): 1},  # M5: A11 + A12
        {(1,0): 1, (0,0): -1}, # M6: A21 - A11
        {(0,1): 1, (1,1): -1}, # M7: A12 - A22
    ]
    
    coeff_y = [
        {(0,0): 1, (1,1): 1},  # M1: B11 + B22
        {(0,0): 1},            # M2: B11
        {(0,1): 1, (1,1): -1}, # M3: B12 - B22
        {(1,0): 1, (0,0): -1}, # M4: B21 - B11
        {(1,1): 1},            # M5: B22
        {(0,0): 1, (0,1): 1},  # M6: B11 + B12
        {(1,0): 1, (1,1): 1},  # M7: B21 + B22
    ]
    
    # C11 = M1 - M4 + M6 + M7
    # C12 = M2 + M3
    # C21 = M5 + M4
    # C22 = M1 - M2 + M3 + M5
    coeff_c = [
        {(0,0): 1},   # M1 -> C11
        {(0,1): 1},   # M2 -> C12
        {(0,1): 1},   # M3 -> C12
        {(0,0): -1, (1,0): 1},  # M4 -> -C11 + C21
        {(1,0): 1},   # M5 -> C21
        {(0,0): 1},   # M6 -> C11
        {(0,0): 1, (1,1): 1},  # M7 -> C11 + C22
    ]
    
    # Wait, M7 contributes to both C11 and C22. Let me recheck.
    # C11 = M1 - M4 + M6 + M7
    # C22 = M1 - M2 + M3 + M5
    #
    # So M7 only goes to C11? No:
    # C11 = M1 - M4 + M6 + M7
    # C12 = M2 + M3
    # C21 = M5 + M4
    # C22 = M1 - M2 + M3 + M5
    #
    # M7 is only in C11. Let me re-examine Strassen's formula.
    #
    # Standard Strassen:
    # C11 = M1 - M4 + M6 + M7
    # C12 = M2 + M3
    # C21 = M5 + M4
    # C22 = M1 - M2 + M3 + M5
    #
    # So M7 is only in C11. That's correct.
    
    coeff_c = [
        {(0,0): 1},                        # M1 -> C11
        {(0,1): 1},                        # M2 -> C12
        {(0,1): 1},                        # M3 -> C12
        {(0,0): -1, (1,0): 1},             # M4 -> -C11 + C21
        {(1,0): 1},                        # M5 -> C21
        {(0,0): 1},                        # M6 -> C11
        {(0,0): 1},                        # M7 -> C11
    ]
    
    # Now build U, V, W
    for r in range(7):
        base_col = r * 27
        for i_prime in range(3):
            for j_prime in range(3):
                for k_prime in range(3):
                    col = base_col + i_prime * 9 + j_prime * 3 + k_prime
                    # For each block (bi, bj) with non-zero coeff_x[r][bi][bj]:
                    for (bi_x, bj_x), cx in coeff_x[r].items():
                        if cx == 0:
                            continue
                        i_row = bi_x * 3 + i_prime
                        j_col = bj_x * 3 + j_prime
                        U[i_row * 6 + j_col][col] += cx
                    
                    for (bi_y, bj_y), cy in coeff_y[r].items():
                        if cy == 0:
                            continue
                        j_row = bi_y * 3 + j_prime
                        k_col = bj_y * 3 + k_prime
                        V[j_row * 6 + k_col][col] += cy
                    
                    for (bi_c, bj_c), cc in coeff_c[r].items():
                        if cc == 0:
                            continue
                        k_row = bi_c * 3 + k_prime
                        i_col = bi_c * 3 + i_prime
                        # W[k*n+i, r] = W[k_row * 6 + i_col, col]
                        W[k_row * 6 + i_col][col] += cc
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
