"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 65 decomposition:
    # 1. 4x4x4 part (columns 0-3 of B, 0-3 of C) using Strassen (Rank 49)
    # 2. 4x4x1 part (column 4 of B, column 4 of C) using Schoolbook (Rank 16)
    
    rank_total = 65
    U = [[0.0] * rank_total for _ in range(n * m)]
    V = [[0.0] * rank_total for _ in range(m * p)]
    W = [[0.0] * rank_total for _ in range(p * n)]
    
    # Base Strassen 2x2x2 decomposition (Rank 7)
    # A: 2x2, B: 2x2, C: 2x2
    # Indices: A: 0,1,2,3 (i*2+j), B: 0,1,2,3 (j*2+k), C: 0,1,2,3 (k*2+i)
    U2 = [
        [1, 0, 0, 1],  # m0
        [0, 0, 1, 1],  # m1
        [1, 0, 0, 0],  # m2
        [0, 0, 0, 1],  # m3
        [1, 1, 0, 0],  # m4
        [1, 0, 1, 0],  # m5
        [0, 1, 0, 1]   # m6
    ]
    V2 = [
        [1, 0, 0, 1],  # m0
        [1, 0, 0, 0],  # m1
        [0, 1, 0, 1],  # m2
        [0, 0, 1, 1],  # m3
        [0, 0, 0, 1],  # m4
        [1, 1, 0, 0],  # m5
        [0, 0, 1, 1]   # m6
    ]
    W2 = [
        [1, 0, 0, 1],   # m0: c00+1, c11+1
        [0, 0, 1, 0],   # m1: c10+1
        [0, 1, 0, 1],   # m2: c01+1, c11+1
        [1, 0, 1, 0],   # m3: c00+1, c10+1
        [-1, 1, 0, 0],  # m4: c00-1, c01+1
        [0, 0, 0, 1],   # m5: c11+1
        [1, 0, 0, 0]    # m6: c00+1
    ]
    
    # Strassen 4x4x4 decomposition
    # 7 macro-products M_s (s=0..6)
    # Each M_s is a 2x2 matrix product.
    # A_s = alpha_11 A_11 + alpha_12 A_12 + alpha_21 A_21 + alpha_22 A_22
    # B_s = beta_11 B_11 + beta_12 B_12 + beta_21 B_21 + beta_22 B_22
    # C blocks get contributions from M_s with coefficients gamma_ij_s
    
    # Coefficients for A_s (rows are s, cols are blocks 11,12,21,22)
    A_coeffs = [
        [1, 0, 0, 1],  # s=0: A11+A22
        [0, 0, 1, 1],  # s=1: A21+A22
        [1, 0, 0, 0],  # s=2: A11
        [0, 0, 0, 1],  # s=3: A22
        [1, 1, 0, 0],  # s=4: A11+A12
        [1, 0, 1, 0],  # s=5: A21-A11 -> -A11 + A21
        [0, 1, 0, -1]  # s=6: A12-A22 -> A12 - A22
    ]
    # Wait, Strassen formula check:
    # M5 = (A21 - A11)(B11 + B12). So A_s for s=5 is -A11 + A21.
    # My list: [1, 0, 1, 0] means 1*A11 + 0*A12 + 1*A21 + 0*A22. This is A11+A21.
    # Correct is -A11 + A21. So [-1, 0, 1, 0].
    # M6 = (A12 - A22)(B21 + B22). So A_s for s=6 is A12 - A22.
    # My list: [0, 1, 0, -1]. This is A12 - A22. Correct.
    
    A_coeffs = [
        [1, 0, 0, 1],  # s=0
        [0, 0, 1, 1],  # s=1
        [1, 0, 0, 0],  # s=2
        [0, 0, 0, 1],  # s=3
        [1, 1, 0, 0],  # s=4
        [-1, 0, 1, 0], # s=5
        [0, 1, 0, -1]  # s=6
    ]
    
    # Coefficients for B_s
    B_coeffs = [
        [1, 0, 0, 1],  # s=0: B11+B22
        [1, 0, 0, 0],  # s=1: B11
        [0, 1, 0, -1], # s=2: B12-B22
        [0, 0, 1, -1], # s=3: B21-B11
        [0, 0, 0, 1],  # s=4: B22
        [1, 1, 0, 0],  # s=5: B11+B12
        [0, 0, 1, 1]   # s=6: B21+B22
    ]
    
    # Coefficients for C blocks (gamma)
    # C11 = M0 + M3 - M4 + M6
    # C12 = M2 + M4
    # C21 = M1 + M3
    # C22 = M0 - M1 + M2 + M5
    # gamma[i_block][j_block][s]
    # Blocks: 11, 12, 21, 22
    C_coeffs = {
        (0, 0): [1, 0, 0, 1, -1, 0, 1],  # C11
        (0, 1): [0, 0, 1, 0, 1, 0, 0],   # C12
        (1, 0): [0, 1, 0, 1, 0, 0, 0],   # C21
        (1, 1): [1, -1, 1, 0, 0, 1, 0]   # C22
    }
    
    # Block index mapping to flat indices in 4x4 matrix
    # Block (br, bc) where br, bc in {0,1}
    # A, B, C1 (4x4 part)
    # A index: i*4+j. Block (br, bc) covers i in br*2..br*2+1, j in bc*2..bc*2+1.
    # A_11: i=0,1 j=0,1 -> 0,1,4,5
    # A_12: i=0,1 j=2,3 -> 2,3,6,7
    # A_21: i=2,3 j=0,1 -> 8,9,12,13
    # A_22: i=2,3 j=2,3 -> 10,11,14,15
    
    def block_indices(br, bc):
        # Returns list of 4 indices for the 2x2 block at row br, col bc in a 4x4 matrix
        # Row-major order
        start_i = br * 2
        start_j = bc * 2
        return [
            start_i * 4 + start_j,
            start_i * 4 + start_j + 1,
            (start_i + 1) * 4 + start_j,
            (start_i + 1) * 4 + start_j + 1
        ]
    
    # W (Output C) index mapping for the 4x4 part (cols 0-3)
    # C index: k*4+i.
    # Block (bk, bi) where bk is row block of C (k), bi is col block of C (i).
    # C_11: k=0,1 i=0,1 -> 0,1,4,5
    # C_12: k=0,1 i=2,3 -> 2,3,6,7
    # C_21: k=2,3 i=0,1 -> 8,9,12,13
    # C_22: k=2,3 i=2,3 -> 10,11,14,15
    # Note: C is 4x5. The 4x4 part uses k=0..3, i=0..3.
    # Indices are k*4+i.
    # Block (bk, bi) covers k in bk*2..bk*2+1, i in bi*2..bi*2+1.
    
    def c_block_indices(bk, bi):
        start_k = bk * 2
        start_i = bi * 2
        return [
            start_k * 4 + start_i,
            start_k * 4 + start_i + 1,
            (start_k + 1) * 4 + start_i,
            (start_k + 1) * 4 + start_i + 1
        ]
    
    col = 0
    for s in range(7):
        for t in range(7):
            # Term for M_s, base term t
            # U vector
            for b_idx in range(4):
                br = b_idx // 2
                bc = b_idx % 2
                coeff = A_coeffs[s][b_idx]
                if coeff != 0:
                    idxs = block_indices(br, bc)
                    for idx in idxs:
                        # U2[t] is the coefficient for the t-th term of the base Strassen
                        # The base term t has value U2[t][local_idx]
                        # We need to map local_idx (0..3) to the block's local positions
                        # The block indices are in row-major order of the 2x2 block.
                        # U2[t] is also in row-major order of the 2x2 matrix.
                        for l in range(4):
                            U[idx][col] += coeff * U2[t][l]
            
            # V vector
            for b_idx in range(4):
                br = b_idx // 2
                bc = b_idx % 2
                coeff = B_coeffs[s][b_idx]
                if coeff != 0:
                    idxs = block_indices(br, bc)
                    for l in range(4):
                        V[idxs[l]][col] += coeff * V2[t][l]
            
            # W vector
            for bk in range(2):
                for bi in range(2):
                    gamma = C_coeffs[(bk, bi)][s]
                    if gamma != 0:
                        idxs = c_block_indices(bk, bi)
                        for l in range(4):
                            W[idxs[l]][col] += gamma * W2[t][l]
            
            col += 1
    
    # 4x4x1 part (Rank 16)
    # A (4x4) * B2 (4x1) = C2 (4x1)
    # C2[i] = sum_j A[i,j] B2[j]
    # B2 is column 4 of B. Index in B: j*5+4.
    # C2 is column 4 of C. Index in C: 4*4+i = 16+i.
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1.0
            V[j * 5 + 4][col] = 1.0
            W[16 + i][col] = 1.0
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
