"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen algorithm for 2x2 matrix multiplication
    # Inputs: A (2x2), B (2x2)
    # Outputs: C (2x2)
    # 7 multiplications:
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = P1 + P4 - P5 + P6
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P3 + P7
    
    # For 4x4, we use block Strassen with 2x2 blocks
    # The 4x4 matrices are partitioned into four 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # Each block multiplication (e.g., A00 * B00) is a 2x2 * 2x2 -> 2x2 multiplication
    # which itself can use Strassen with 7 scalar multiplications.
    #
    # Total rank = 7 (outer Strassen) * 7 (inner Strassen per block) = 49
    # Wait, that's not quite right. Let me think more carefully.
    #
    # Actually, in the block Strassen approach for 4x4:
    # We have 7 "multiplications" at the block level, each involving sums/differences
    # of 2x2 blocks. Each such "block multiplication" requires computing the product
    # of two 2x2 matrices, which via Strassen takes 7 scalar multiplications.
    # So total = 7 * 7 = 49.
    
    # Let's build this explicitly.
    # 
    # A is a 4x4 matrix, indexed as A[i][j] for i,j in 0..3
    # We flatten A to a vector of length 16: index = i*4 + j
    # Similarly for B: index = j*4 + k (where j is row, k is col)
    # C is output: index = k*4 + i (per the convention W[k*n+i, r])
    # Wait, the convention is W[k*n+i, r], so W has p*n = 16 rows.
    # The output C[k][i] corresponds to W[k*n+i].
    #
    # For matrix multiplication: C = A * B
    # C[i][k] = sum_j A[i][j] * B[j][k]
    # But the convention says W[k*n+i], so the output element at position (k, i) in the
    # flattened W corresponds to C[i][k]? Let me re-read.
    #
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # In matrix multiplication C = A*B, C[i][k] = sum_j A[i][j]*B[j][k]
    # If W[k*n+i] represents C[i][k], then the index is k*n + i where i is the row of C
    # and k is the column of C. So W[row_C * n + col_C]? No, k*n+i with k first.
    #
    # Let's just follow the convention: W has shape (p*n, rank). The index is k*n + i.
    # In C = A * B, the (i,k)-th entry of C is sum_j A[i,j] * B[j,k].
    # So W[k*n + i, r] should capture the contribution to C[i][k].
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Strassen for 2x2:
    # A2 = [[a00, a01], [a10, a11]] (indices 0,1,2,3 in flattened 2x2)
    # B2 = [[b00, b01], [b10, b11]] (indices 0,1,2,3 in flattened 2x2)
    # C2[i][j] = sum_k A2[i][k] * B2[k][j]
    #
    # Strassen formulas for 2x2:
    # P1 = (a00 + a11) * (b00 + b11)
    # P2 = (a10 + a11) * b00
    # P3 = a00 * (b01 - b11)
    # P4 = a11 * (b10 - b00)
    # P5 = (a00 + a01) * b11
    # P6 = (a10 - a00) * (b00 + b01)
    # P7 = (a01 - a11) * (b10 + b11)
    #
    # C00 = P1 + P4 - P5 + P6
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P3 + P7
    
    # For each of the 7 block-level Strassen products, we need to:
    # 1. Extract the relevant 2x2 block from A (with coefficients)
    # 2. Extract the relevant 2x2 block from B (with coefficients)
    # 3. Multiply them using 2x2 Strassen (7 scalar mults)
    # 4. Add the result to the appropriate 2x2 block of C (with coefficients)
    
    # Let's define the block-level Strassen for 4x4 with 2x2 blocks.
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3)
    #            A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3)
    #            B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    # C blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3)
    #            C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3)
    
    # Block-level Strassen:
    # P1_block = (A00 + A11) * (B00 + B11)
    # P2_block = (A10 + A11) * B00
    # P3_block = A00 * (B01 - B11)
    # P4_block = A11 * (B10 - B00)
    # P5_block = (A00 + A01) * B11
    # P6_block = (A10 - A00) * (B00 + B01)
    # P7_block = (A01 - A11) * (B10 + B11)
    #
    # C00 = P1_block + P4_block - P5_block + P6_block
    # C01 = P3_block + P5_block
    # C10 = P2_block + P4_block
    # C11 = P1_block - P3_block + P7_block
    
    # For each block product, e.g., (A00 + A11) * (B00 + B11):
    # The left factor is a 2x2 matrix: (A00 + A11)[i][j] = A00[i][j] + A11[i][j]
    # The right factor is a 2x2 matrix: (B00 + B11)[i][j] = B00[i][j] + B11[i][j]
    # Their product is a 2x2 matrix, computed via 2x2 Strassen.
    
    # Let's enumerate all block products and their contributions.
    
    # Helper: get block indices
    # A[i][j] in 4x4: global index = i*4 + j
    # Block A00: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # Block A01: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # Block A10: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # Block A11: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # B[j][k] in 4x4: global index = j*4 + k
    # Block B00: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # Block B01: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # Block B10: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # Block B11: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # C[i][k] in 4x4: W[k*4 + i]
    # Block C00: rows 0-1, cols 0-1 -> W indices 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # Block C01: rows 0-1, cols 2-3 -> W indices 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # Block C10: rows 2-3, cols 0-1 -> W indices 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # Block C11: rows 2-3, cols 2-3 -> W indices 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    
    # Define block row/col ranges for 2x2 blocks within 4x4:
    # Block (bi, bj) where bi, bj in {0,1}:
    #   A rows: bi*2, bi*2+1; A cols: bj*2, bj*2+1
    #   A global indices: for row r in [bi*2, bi*2+1], col c in [bj*2, bj*2+1]: r*4+c
    
    def a_block_indices(bi, bj):
        """Return list of 4 global indices for A block (bi, bj)."""
        rows = [bi*2, bi*2+1]
        cols = [bj*2, bj*2+1]
        return [r*4 + c for r in rows for c in cols]
    
    def b_block_indices(bi, bj):
        """Return list of 4 global indices for B block (bi, bj)."""
        rows = [bi*2, bi*2+1]
        cols = [bj*2, bj*2+1]
        return [r*4 + c for r in rows for c in cols]
    
    def c_block_indices(bi, bj):
        """Return list of 4 W indices for C block (bi, bj).
        C[i][k] -> W[k*4 + i]
        Block (bi, bj): rows i in [bi*2, bi*2+1], cols k in [bj*2, bj*2+1]
        """
        rows = [bi*2, bi*2+1]
        cols = [bj*2, bj*2+1]
        return [k*4 + i for i in rows for k in cols]
    
    # 2x2 Strassen: for multiplying two 2x2 matrices L and R to get product P
    # L and R are represented by their 4 elements in row-major order.
    # We need to express each of the 7 Strassen products as linear combinations
    # of L and R elements, and the output P elements as linear combinations of the 7 products.
    
    # For 2x2 Strassen:
    # L = [[l0, l1], [l2, l3]] (l0=L[0][0], l1=L[0][1], l2=L[1][0], l3=L[1][1])
    # R = [[r0, r1], [r2, r3]] (r0=R[0][0], r1=R[0][1], r2=R[1][0], r3=R[1][1])
    
    # P1 = (l0 + l3) * (r0 + r3)
    # P2 = (l2 + l3) * r0
    # P3 = l0 * (r1 - r3)
    # P4 = l3 * (r2 - r0)
    # P5 = (l0 + l1) * r3
    # P6 = (l2 - l0) * (r0 + r1)
    # P7 = (l1 - l3) * (r2 + r3)
    
    # Output:
    # P[0][0] = P1 + P4 - P5 + P6
    # P[0][1] = P3 + P5
    # P[1][0] = P2 + P4
    # P[1][1] = P1 - P3 + P7
    
    # Now, for each of the 7 block-level Strassen products, we have:
    # A linear combination of A blocks (with coefficients) multiplied by
    # a linear combination of B blocks (with coefficients).
    # Each such product involves 7 scalar Strassen multiplications.
    
    # Let's define the block-level Strassen coefficients.
    # For block product q (0-6), we have:
    #   L = sum_{(bi,bj)} coeff_A[q][(bi,bj)] * A_block(bi,bj)
    #   R = sum_{(bi,bj)} coeff_B[q][(bi,bj)] * B_block(bi,bj)
    #   P = L * R (2x2 matrix product via Strassen)
    #   C += sum_{(ci,cj)} coeff_C[q][(ci,cj)] * P_block(ci,cj)
    
    # Block-level Strassen for 4x4 with 2x2 blocks:
    # q=0: L = A00 + A11, R = B00 + B11
    # q=1: L = A10 + A11, R = B00
    # q=2: L = A00, R = B01 - B11
    # q=3: L = A11, R = B10 - B00
    # q=4: L = A00 + A01, R = B11
    # q=5: L = A10 - A00, R = B00 + B01
    # q=6: L = A01 - A11, R = B10 + B11
    
    # Output:
    # C00 = P0 + P3 - P4 + P5
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P2 + P6
    
    # Let's encode this.
    
    # coeff_A[q] is a dict: (bi,bj) -> coefficient
    coeff_A = [
        {(0,0): 1, (1,1): 1},  # q=0
        {(1,0): 1, (1,1): 1},  # q=1
        {(0,0): 1},             # q=2
        {(1,1): 1},             # q=3
        {(0,0): 1, (0,1): 1},  # q=4
        {(1,0): 1, (0,0): -1}, # q=5
        {(0,1): 1, (1,1): -1}, # q=6
    ]
    
    coeff_B = [
        {(0,0): 1, (1,1): 1},  # q=0
        {(0,0): 1},             # q=1
        {(0,1): 1, (1,1): -1}, # q=2
        {(1,0): 1, (0,0): -1}, # q=3
        {(1,1): 1},             # q=4
        {(0,0): 1, (0,1): 1},  # q=5
        {(1,0): 1, (1,1): 1},  # q=6
    ]
    
    coeff_C = [
        {(0,0): 1, (1,1): 1},  # q=0 contributes to C00 and C11
        {(1,0): 1},             # q=1 contributes to C10
        {(0,1): 1},             # q=2 contributes to C01
        {(0,0): 1, (1,0): 1},  # q=3 contributes to C00 and C10
        {(0,0): -1, (0,1): 1}, # q=4 contributes to C00 and C01
        {(0,0): 1},             # q=5 contributes to C00
        {(1,1): 1},             # q=6 contributes to C11
    ]
    
    # Wait, let me double-check the output formulas:
    # C00 = P0 + P3 - P4 + P5
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P2 + P6
    
    # So:
    # C00: q=0: +1, q=3: +1, q=4: -1, q=5: +1
    # C01: q=2: +1, q=4: +1
    # C10: q=1: +1, q=3: +1
    # C11: q=0: +1, q=2: -1, q=6: +1
    
    coeff_C = [
        {(0,0): 1},             # q=0 -> C00
        {(1,0): 1},             # q=1 -> C10
        {(0,1): 1, (1,1): -1}, # q=2 -> C01 (+1), C11 (-1)
        {(0,0): 1, (1,0): 1},  # q=3 -> C00 (+1), C10 (+1)
        {(0,0): -1, (0,1): 1}, # q=4 -> C00 (-1), C01 (+1)
        {(0,0): 1},             # q=5 -> C00
        {(1,1): 1},             # q=6 -> C11
    ]
    
    # Now, for each block product q (0-6) and each inner Strassen product s (0-6),
    # we have a scalar multiplication. The total rank is 7*7=49.
    
    # For inner Strassen s, the left factor L is a linear combination of the 4 elements
    # of the 2x2 block L_block, and the right factor R is a linear combination of the 4
    # elements of the 2x2 block R_block.
    
    # Inner Strassen definitions:
    # s=0: L_inner = l0 + l3, R_inner = r0 + r3
    # s=1: L_inner = l2 + l3, R_inner = r0
    # s=2: L_inner = l0, R_inner = r1 - r3
    # s=3: L_inner = l3, R_inner = r2 - r0
    # s=4: L_inner = l0 + l1, R_inner = r3
    # s=5: L_inner = l2 - l0, R_inner = r0 + r1
    # s=6: L_inner = l1 - l3, R_inner = r2 + r3
    
    # Output:
    # P[0][0] = s0 + s3 - s4 + s5
    # P[0][1] = s2 + s4
    # P[1][0] = s1 + s3
    # P[1][1] = s0 - s2 + s6
    
    # For each (q, s), we need to determine:
    # - Which A elements contribute to L_inner, and with what coefficients
    # - Which B elements contribute to R_inner, and with what coefficients
    # - Which C elements receive the product, and with what coefficients
    
    # The L_block for product q is: L_block[i][j] = sum_{(bi,bj)} coeff_A[q][(bi,bj)] * A[bi*2+i, bj*2+j]
    # The R_block for product q is: R_block[i][j] = sum_{(bi,bj)} coeff_B[q][(bi,bj)] * B[bi*2+i, bj*2+j]
    
    # For inner Strassen s, L_inner is a linear combination of L_block elements:
    # l0 = L_block[0][0], l1 = L_block[0][1], l2 = L_block[1][0], l3 = L_block[1][1]
    
    # Similarly for R_inner with r0, r1, r2, r3.
    
    # Let's build the U, V, W matrices.
    
    # For each q in 0..6, s in 0..6:
    #   rank_idx = q * 7 + s
    #   
    #   # Determine L_inner coefficients for l0, l1, l2, l3
    #   if s == 0:
    #       l_coeffs = [1, 0, 0, 1]  # l0 + l3
    #   elif s == 1:
    #       l_coeffs = [0, 0, 1, 1]  # l2 + l3
    #   elif s == 2:
    #       l_coeffs = [1, 0, 0, 0]  # l0
    #   elif s == 3:
    #       l_coeffs = [0, 0, 0, 1]  # l3
    #   elif s == 4:
    #       l_coeffs = [1, 1, 0, 0]  # l0 + l1
    #   elif s == 5:
    #       l_coeffs = [-1, 0, 1, 0] # l2 - l0
    #   elif s == 6:
    #       l_coeffs = [0, 1, 0, -1] # l1 - l3
    #   
    #   # Determine R_inner coefficients for r0, r1, r2, r3
    #   if s == 0:
    #       r_coeffs = [1, 0, 0, 1]  # r0 + r3
    #   elif s == 1:
    #       r_coeffs = [1, 0, 0, 0]  # r0
    #   elif s == 2:
    #       r_coeffs = [0, 1, 0, -1] # r1 - r3
    #   elif s == 3:
    #       r_coeffs = [0, 0, 1, -1] # r2 - r0
    #   elif s == 4:
    #       r_coeffs = [0, 0, 0, 1]  # r3
    #   elif s == 5:
    #       r_coeffs = [1, 1, 0, 0]  # r0 + r1
    #   elif s == 6:
    #       r_coeffs = [0, 0, 1, 1]  # r2 + r3
    
    l_inner_coeffs = [
        [1, 0, 0, 1],  # s=0
        [0, 0, 1, 1],  # s=1
        [1, 0, 0, 0],  # s=2
        [0, 0, 0, 1],  # s=3
        [1, 1, 0, 0],  # s=4
        [-1, 0, 1, 0], # s=5
        [0, 1, 0, -1], # s=6
    ]
    
    r_inner_coeffs = [
        [1, 0, 0, 1],  # s=0
        [1, 0, 0, 0],  # s=1
        [0, 1, 0, -1], # s=2
        [0, 0, 1, -1], # s=3
        [0, 0, 0, 1],  # s=4
        [1, 1, 0, 0],  # s=5
        [0, 0, 1, 1],  # s=6
    ]
    
    # Output coefficients for C_block elements:
    # p_out_coeffs[s][ci][cj] where ci, cj in {0,1}
    # P[0][0] = s0 + s3 - s4 + s5
    # P[0][1] = s2 + s4
    # P[1][0] = s1 + s3
    # P[1][1] = s0 - s2 + s6
    
    p_out_coeffs = [
        [[1, 0], [0, 0]],  # s=0: P[0][0]
        [[0, 0], [1, 0]],  # s=1: P[1][0]
        [[0, 1], [0, 0]],  # s=2: P[0][1]
        [[1, 0], [1, 0]],  # s=3: P[0][0], P[1][0]
        [[-1, 1], [0, 0]], # s=4: P[0][0], P[0][1]
        [[1, 0], [0, 0]],  # s=5: P[0][0]
        [[0, 0], [0, 1]],  # s=6: P[1][1]
    ]
    
    # Now iterate over all (q, s) pairs.
    for q in range(7):
        for s in range(7):
            rank_idx = q * 7 + s
            
            # Get the inner Strassen coefficients
            lc = l_inner_coeffs[s]  # [l0, l1, l2, l3]
            rc = r_inner_coeffs[s]  # [r0, r1, r2, r3]
            
            # For each A block (bi, bj) in coeff_A[q]:
            #   For each element (i, j) in 2x2:
            #     L_block[i][j] contribution from A[bi*2+i, bj*2+j] is coeff_A[q][(bi,bj)]
            #     The coefficient for l_k (where k=0..3 corresponds to (i,j) in order) is lc[k]
            #     So A global index = (bi*2+i)*4 + (bj*2+j)
            #     U[A_global_idx, rank_idx] += coeff_A[q][(bi,bj)] * lc[k]
            
            # Map (i,j) in 2x2 to index k: k = i*2 + j
            for (bi, bj), a_coeff in coeff_A[q].items():
                if a_coeff == 0:
                    continue
                for i in range(2):
                    for j in range(2):
                        k = i * 2 + j
                        a_global = (bi * 2 + i) * 4 + (bj * 2 + j)
                        U[a_global][rank_idx] += a_coeff * lc[k]
            
            # Similarly for B:
            for (bi, bj), b_coeff in coeff_B[q].items():
                if b_coeff == 0:
                    continue
                for i in range(2):
                    for j in range(2):
                        k = i * 2 + j
                        b_global = (bi * 2 + i) * 4 + (bj * 2 + j)
                        V[b_global][rank_idx] += b_coeff * rc[k]
            
            # For C:
            # The product P is a 2x2 matrix. P[pi][pj] gets contribution from s.
            # p_out_coeffs[s][pi][pj] gives the coefficient.
            # This P block is then added to C blocks with coeff_C[q].
            # For each (ci, cj) in coeff_C[q]:
            #   C_block(ci, cj) = sum over (pi, pj) ... wait, no.
            #   The product L*R gives a 2x2 matrix P.
            #   P[pi][pj] is the (pi, pj)-th element of the product.
            #   This P block is placed at C position (ci, cj) with coefficient coeff_C[q][(ci,cj)].
            #   So C[ci*2+pi, cj*2+pj] gets contribution.
            #   W index = (cj*2+pj)*4 + (ci*2+pi)
            
            for (ci, cj), c_coeff in coeff_C[q].items():
                if c_coeff == 0:
                    continue
                for pi in range(2):
                    for pj in range(2):
                        p_coeff = p_out_coeffs[s][pi][pj]
                        if p_coeff == 0:
                            continue
                        c_row = ci * 2 + pi
                        c_col = cj * 2 + pj
                        w_idx = c_col * 4 + c_row
                        W[w_idx][rank_idx] += c_coeff * p_coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

