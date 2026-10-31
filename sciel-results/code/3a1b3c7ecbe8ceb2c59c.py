"""Optimal 4x4x4 matrix multiplication tensor decomposition using Strassen's algorithm (rank 49)."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-49 exact tensor decomposition for 4x4 matrix multiplication
    using Strassen's algorithm.
    
    The factor matrices have shapes:
      U: (16, 49) - corresponds to n*m = 4*4
      V: (16, 49) - corresponds to m*p = 4*4  
      W: (16, 49) - corresponds to p*n = 4*4
    
    Entries are integers (0 or +/-1).
    Indexing convention:
      U[i*m + j, r] for i in [0,n), j in [0,m), r in [0,rank)
      V[j*p + k, r] for j in [0,m), k in [0,p), r in [0,rank)
      W[k*n + i, r] for k in [0,p), i in [0,n), r in [0,rank)
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    rank = 49
    
    # Initialize factor matrices as lists of lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We'll build the decomposition by composing two levels of Strassen.
    # Level 1: Split 4x4 into 2x2 blocks, apply Strassen (7 multiplications at this level)
    # Level 2: Each 2x2x2 multiplication uses Strassen (7 multiplications)
    # Total: 7 * 7 = 49 scalar multiplications
    
    # Strassen's 7 intermediate products for 2x2 matrices:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # Result C:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4, we index blocks as (block_i, block_j) where block_i, block_j in {0, 1}
    # Block (bi, bj) contains elements A[2*bi : 2*bi+2, 2*bj : 2*bj+2]
    #
    # At the top level (4x4), we have 4 sub-blocks:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x2
    #
    # Strassen at top level:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mj is a 2x2x2 tensor (product of two 2x2 matrices), which we decompose
    # using Strassen's 7-multiplication scheme.
    
    # Let's define the Strassen coefficients for a 2x2x2 multiplication.
    # For each of the 7 intermediate products, we specify which input indices
    # (i, j) for U and (j, k) for V are used, and with what sign.
    
    # Strassen 2x2x2: For product C = AB where A, B, C are 2x2:
    # M1 = (A[0,0] + A[1,1])(B[0,0] + B[1,1])
    #   U terms: A[0,0] with coeff +1, A[1,1] with coeff +1
    #   V terms: B[0,0] with coeff +1, B[1,1] with coeff +1
    #   W terms: C[0,0] +1, C[0,1] -1, C[1,0] -1, C[1,1] +1
    #
    # Wait, let me be more careful. In tensor decomposition form:
    # C[i,k] = sum_r U[i,j_r] * V[j_r,k] * W[i,k,r] ... no, that's not right.
    #
    # The correct form is: C[i,k] = sum_{j=0}^{m-1} sum_r U[i*m+j, r] * V[j*p+k, r] * ...
    # Actually, for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor decomposition: C[i,k] = sum_r U[i*m+j_r, r] * V[j_r*p+k, r]
    # where for each r, j_r is fixed, and U and V have entries that encode the
    # linear combinations.
    #
    # More precisely: For each rank component r:
    #   U has non-zero entries at positions (i, j_r) for various i
    #   V has non-zero entries at positions (j_r, k) for various k  
    #   W has non-zero entries at positions (k, i) for various k, i
    #   The product U[i*m+j_r, r] * V[j_r*p+k, r] * W[k*n+i, r] contributes to C[i,k]
    #
    # Actually, the standard tensor decomposition form is:
    # C[i,k] = sum_r U[i*m + j_r, r] * V[j_r*p + k, r]
    # where j_r is implicitly determined by which rows/columns are non-zero in U and V.
    #
    # Let me think about this differently. For each scalar multiplication in the
    # algorithm, we create one rank component. For Strassen 2x2x2:
    
    # I'll build the decomposition by iterating over all 49 rank components.
    # For each component, I need to determine which elements of A, B, and C it involves.
    
    # Top-level Strassen: 7 intermediate products M1-M7
    # Each Mj involves a 2x2x2 Strassen decomposition with 7 sub-products
    # Total: 7 * 7 = 49
    
    # Let me define the top-level Strassen terms.
    # For term t (0-6), the A-blocks involved and their coefficients:
    top_A_coeffs = [
        # M1: (A00 + A11) -> A00: +1, A11: +1
        {0: 1, 3: 1},
        # M2: (A10 + A11) -> A10: +1, A11: +1  
        {2: 1, 3: 1},
        # M3: A00 -> A00: +1
        {0: 1},
        # M4: A11 -> A11: +1
        {3: 1},
        # M5: (A00 + A01) -> A00: +1, A01: +1
        {0: 1, 1: 1},
        # M6: (A10 - A00) -> A10: +1, A00: -1
        {2: 1, 0: -1},
        # M7: (A01 - A11) -> A01: +1, A11: -1
        {1: 1, 3: -1},
    ]
    
    # For term t (0-6), the B-blocks involved and their coefficients:
    top_B_coeffs = [
        # M1: (B00 + B11) -> B00: +1, B11: +1
        {0: 1, 3: 1},
        # M2: B00 -> B00: +1
        {0: 1},
        # M3: (B10 - B11) -> B10: +1, B11: -1
        {2: 1, 3: -1},
        # M4: (B01 - B00) -> B01: +1, B00: -1
        {1: 1, 0: -1},
        # M5: B11 -> B11: +1
        {3: 1},
        # M6: (B00 + B01) -> B00: +1, B01: +1
        {0: 1, 1: 1},
        # M7: (B10 + B11) -> B10: +1, B11: +1
        {2: 1, 3: 1},
    ]
    
    # For term t (0-6), the C-blocks that receive this term's result, with coefficients:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    top_C_coeffs = [
        # M1 contributes to: C00: +1, C11: +1
        {0: 1, 3: 1},
        # M2 contributes to: C10: +1, C11: -1
        {2: 1, 3: -1},
        # M3 contributes to: C01: +1, C11: +1
        {1: 1, 3: 1},
        # M4 contributes to: C00: +1, C10: +1
        {0: 1, 2: 1},
        # M5 contributes to: C00: -1, C01: +1
        {0: -1, 1: 1},
        # M6 contributes to: C11: +1
        {3: 1},
        # M7 contributes to: C00: +1
        {0: 1},
    ]
    
    # Now for the inner 2x2x2 Strassen. For each inner term s (0-6):
    # The inner Strassen computes a 2x2 product from two 2x2 matrices.
    # The inner A-block is a 2x2 block, with elements at positions (2*bi + ii, 2*bj + jj)
    # where bi, bj are the block indices and ii, jj are within-block indices.
    
    # Inner Strassen 2x2x2 terms:
    # For a 2x2 matrix X with elements X[0,0], X[0,1], X[1,0], X[1,1]:
    inner_A_coeffs = [
        # M1: X00 + X11
        {(0, 0): 1, (1, 1): 1},
        # M2: X10 + X11
        {(1, 0): 1, (1, 1): 1},
        # M3: X00
        {(0, 0): 1},
        # M4: X11
        {(1, 1): 1},
        # M5: X00 + X01
        {(0, 0): 1, (0, 1): 1},
        # M6: X10 - X00
        {(1, 0): 1, (0, 0): -1},
        # M7: X01 - X11
        {(0, 1): 1, (1, 1): -1},
    ]
    
    inner_B_coeffs = [
        # M1: Y00 + Y11
        {(0, 0): 1, (1, 1): 1},
        # M2: Y00
        {(0, 0): 1},
        # M3: Y10 - Y11
        {(1, 0): 1, (1, 1): -1},
        # M4: Y01 - Y00
        {(0, 1): 1, (0, 0): -1},
        # M5: Y11
        {(1, 1): 1},
        # M6: Y00 + Y01
        {(0, 0): 1, (0, 1): 1},
        # M7: Y10 + Y11
        {(1, 0): 1, (1, 1): 1},
    ]
    
    inner_C_coeffs = [
        # M1: Z00 +1, Z01 -1, Z10 -1, Z11 +1
        {(0, 0): 1, (0, 1): -1, (1, 0): -1, (1, 1): 1},
        # M2: Z10 +1, Z11 -1
        {(1, 0): 1, (1, 1): -1},
        # M3: Z01 +1, Z11 +1
        {(0, 1): 1, (1, 1): 1},
        # M4: Z00 +1, Z10 +1
        {(0, 0): 1, (1, 0): 1},
        # M5: Z00 -1, Z01 +1
        {(0, 0): -1, (0, 1): 1},
        # M6: Z11 +1
        {(1, 1): 1},
        # M7: Z00 +1
        {(0, 0): 1},
    ]
    
    # Now build the 49 rank components.
    # Rank component index: r = t * 7 + s where t is top-level term (0-6), s is inner term (0-6)
    
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            
            # Determine which top-level A-blocks are involved and their coefficients
            a_blocks = top_A_coeffs[t]  # dict: block_index -> coeff
            # Determine which top-level B-blocks are involved and their coefficients
            b_blocks = top_B_coeffs[t]  # dict: block_index -> coeff
            # Determine which top-level C-blocks receive this term
            c_blocks = top_C_coeffs[t]  # dict: block_index -> coeff
            
            # For the inner Strassen term s:
            # inner_A_coeffs[s]: dict (ii, jj) -> coeff for the A-block
            # inner_B_coeffs[s]: dict (jj, kk) -> coeff for the B-block
            # inner_C_coeffs[s]: dict (kk, ii) -> coeff for the C-block
            
            # For each combination of top A-block and inner A position:
            # The actual matrix element A[2*bi + ii, 2*bj + jj]
            # contributes with coefficient: top_coeff * inner_coeff
            
            # U[i*m + j, r] where i = row, j = col of A
            # For each (a_block, a_pos) in inner_A_coeffs[s] where a_block is in a_blocks:
            for a_block, a_coeff in a_blocks.items():
                bi = a_block // 2  # block row
                bj = a_block % 2   # block col
                for (ii, jj), inner_coeff in inner_A_coeffs[s].items():
                    # A element at (2*bi + ii, 2*bj + jj)
                    i = 2 * bi + ii
                    j = 2 * bj + jj
                    total_coeff = a_coeff * inner_coeff
                    U[i * m + j][r] = total_coeff
            
            # V[j*p + k, r] where j = row of B, k = col of B
            for b_block, b_coeff in b_blocks.items():
                bj2 = b_block // 2  # block row of B
                bk2 = b_block % 2   # block col of B
                for (jj, kk), inner_coeff in inner_B_coeffs[s].items():
                    # B element at (2*bj2 + jj, 2*bk2 + kk)
                    j = 2 * bj2 + jj
                    k = 2 * bk2 + kk
                    total_coeff = b_coeff * inner_coeff
                    V[j * p + k][r] = total_coeff
            
            # W[k*n + i, r] where k = row of C, i = col of C
            for c_block, c_coeff in c_blocks.items():
                ck = c_block // 2  # block row of C
                ci = c_block % 2   # block col of C
                for (kk, ii), inner_coeff in inner_C_coeffs[s].items():
                    # C element at (2*ck + kk, 2*ci + ii)
                    k = 2 * ck + kk
                    i = 2 * ci + ii
                    total_coeff = c_coeff * inner_coeff
                    W[k * n + i][r] = total_coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
