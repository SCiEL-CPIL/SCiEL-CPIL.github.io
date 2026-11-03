"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # We'll compute C = A * B where A is 4x4, B is 4x7
    # Split B into B1 (4x4) and B2 (4x3)
    # C1 = A * B1 (4x4*4x4) using Strassen rank 49
    # C2 = A * B2 (4x4*4x3) using direct rank 48
    # Total rank: 49 + 48 = 97
    
    rank = 97
    U = [[0] * rank for _ in range(n * m)]  # 16 x 97
    V = [[0] * rank for _ in range(m * p)]  # 28 x 97
    W = [[0] * rank for _ in range(p * n)]  # 28 x 97
    
    # Part 1: 4x4 * 4x4 Strassen decomposition (rank 49)
    # A is 4x4, split into 2x2 blocks: A11, A12, A21, A22
    # B1 is 4x4, split into 2x2 blocks: B11, B12, B21, B22
    # C1 = A * B1, split into 2x2 blocks: C11, C12, C21, C22
    
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2*2x2 multiplication with rank 7
    # We need to encode the Strassen structure
    
    # For 2x2*2x2 Strassen, the 7 products are:
    # P1 = (a11+a22)(b11+b22)
    # P2 = (a21+a22)b11
    # P3 = a11(b12-b22)
    # P4 = a22(b21-b11)
    # P5 = (a11+a12)b22
    # P6 = (a21-a11)(b11+b12)
    # P7 = (a12-a22)(b21+b22)
    # c11 = P1 + P4 - P5 + P7
    # c12 = P3 + P5
    # c21 = P2 + P4
    # c22 = P1 - P2 + P3 + P6
    
    # For each 2x2*2x2 block multiplication, we need 7 rank-1 terms
    # Each rank-1 term: U has 1 in two positions, V has 1 in two positions, W has coeffs
    
    # Let me define a helper to add a 2x2*2x2 Strassen block
    # The 2x2 matrix A occupies rows [r0:r0+2, c0:c0+2] in the 4x4 A
    # But in our flattened representation, A[i*m+j] where i,j are 0..3
    # Block A11: i=0,1; j=0,1 -> indices 0,1,4,5
    # Block A12: i=0,1; j=2,3 -> indices 2,3,6,7
    # Block A21: i=2,3; j=0,1 -> indices 8,9,12,13
    # Block A22: i=2,3; j=2,3 -> indices 10,11,14,15
    
    # For B1 (4x4), similar indexing in the first 4 columns of B (4x7)
    # B[j*p+k] where j=0..3, k=0..6
    # B11: j=0,1; k=0,1 -> indices 0,1,4,5
    # B12: j=0,1; k=2,3 -> indices 2,3,6,7
    # B21: j=2,3; k=0,1 -> indices 8,9,12,13
    # B22: j=2,3; k=2,3 -> indices 10,11,14,15
    
    # For C1 (4x4), output C[k*n+i] where k=0..3, i=0..3
    # C11: k=0,1; i=0,1 -> indices 0,1,4,5
    # C12: k=0,1; i=2,3 -> indices 2,3,6,7
    # C21: k=2,3; i=0,1 -> indices 8,9,12,13
    # C22: k=2,3; i=2,3 -> indices 10,11,14,15
    
    # This is getting complex. Let me just build the full 49-rank Strassen for 4x4*4x4.
    
    # Actually, let me just implement it directly by enumerating the 49 terms.
    
    # I'll create a function to add a rank-1 term
    col = 0
    
    def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal col
        for idx, val in zip(u_indices, u_vals):
            U[idx][col] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][col] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][col] += val
        col += 1
    
    # Strassen 4x4*4x4: 7 levels of 2x2*2x2, each with 7 rank-1 terms = 49 terms
    
    # Define the 7 Strassen combinations for 2x2*2x2
    # Each combination: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict from (i,j) to coeff for A block
    # B_coeffs: dict from (j,k) to coeff for B block  
    # C_coeffs: dict from (k,i) to coeff for C block
    
    # For a 2x2*2x2 at block position (bi, bj) for A, (bj, bk) for B, (bk, bi) for C
    # where bi, bj, bk are 0 or 1 (block indices within the 4x4)
    
    # The 7 Strassen products for 2x2*2x2:
    strassen_2x2 = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # P1 = (a00+a11)(b00+b11) -> c00 += P1, c11 += P1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # P2 = (a10+a11)b00 -> c10 += P2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # P3 = a00(b01-b11) -> c01 += P3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
        # P4 = a11(b10-b00) -> c11 -= P4, c10 += P4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(1,1): -1, (1,0): 1}),
        # P5 = (a00+a01)b11 -> c01 -= P5, c11 += P5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): -1, (1,1): 1}),
        # P6 = (a10-a00)(b00+b01) -> c11 += P6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # P7 = (a01-a11)(b10+b11) -> c00 += P7, c11 -= P7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): -1}),
    ]
    
    # Now for each of the 4 block positions in the 4x4*4x4, apply Strassen
    # Block positions: (bi, bk) where bi is row block, bk is col block, each 0 or 1
    # A block: (bi, bj) for bj in 0,1
    # B block: (bj, bk) for bj in 0,1
    # C block: (bk, bi) for bk in 0,1
    
    # For each output block (bi, bk), we sum over bj
    # C(bi,bk) = sum_bj A(bi,bj) * B(bj,bk)
    # Using Strassen: 7 products per (bi,bk,bj) combination? No, Strassen is for 2x2*2x2.
    
    # Actually, for 4x4*4x4, we split into 2x2 blocks and use Strassen on the 2x2 block matrix.
    # The block-level Strassen has 7 block multiplications, each of which is 2x2*2x2.
    # So total 7 * 7 = 49 rank-1 terms.
    
    # Let me define the block-level Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2*2x2 product.
    
    # For each Mi, I need to add 7 rank-1 terms (Strassen for 2x2*2x2)
    # with appropriate coefficients for the A and B blocks, and accumulate into C blocks.
    
    # Let me define the block-level Strassen terms:
    # Each term: (A_block_coeffs, B_block_coeffs, C_block_coeffs)
    # A_block_coeffs: dict from (bi,bj) to coeff
    # B_block_coeffs: dict from (bj,bk) to coeff
    # C_block_coeffs: dict from (bk,bi) to coeff
    
    block_strassen = [
        # M1 = (A11+A22)(B11+B22)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),  # contributes to C11 and C22
        # M2 = (A21+A22)B11
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),  # contributes to C21
        # M3 = A11(B12-B22)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),  # contributes to C12
        # M4 = A22(B21-B11)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(1,0): 1, (0,0): -1}),  # contributes to C21 and C11
        # M5 = (A11+A12)B22
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): -1, (1,1): 1}),  # contributes to C12 and C22
        # M6 = (A21-A11)(B11+B12)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),  # contributes to C22
        # M7 = (A12-A22)(B21+B22)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): -1}),  # contributes to C11 and C22
    ]
    
    # Now for each block Strassen term, and for each of the 7 sub-Strassen terms,
    # add a rank-1 term.
    
    for a_block_coeffs, b_block_coeffs, c_block_coeffs in block_strassen:
        for sub_a_coeffs, sub_b_coeffs, sub_c_coeffs in strassen_2x2:
            u_indices = []
            u_vals = []
            v_indices = []
            v_vals = []
            w_indices = []
            w_vals = []
            
            for (bi, bj), a_coeff in a_block_coeffs.items():
                for (si, sj), sub_coeff in sub_a_coeffs.items():
                    # A index: (bi*2+si, bj*2+sj) -> (bi*2+si)*4 + (bj*2+sj)
                    idx = (bi * 2 + si) * 4 + (bj * 2 + sj)
                    val = a_coeff * sub_coeff
                    if val != 0:
                        u_indices.append(idx)
                        u_vals.append(val)
            
            for (bj, bk), b_coeff in b_block_coeffs.items():
                for (sj, sk), sub_coeff in sub_b_coeffs.items():
                    # B1 index: (bj, bk) in the 4x4 part of B (4x7)
                    # B[j*7 + k] where j = bj*2+sj, k = bk*2+sk
                    idx = (bj * 2 + sj) * 7 + (bk * 2 + sk)
                    val = b_coeff * sub_coeff
                    if val != 0:
                        v_indices.append(idx)
                        v_vals.append(val)
            
            for (bk, bi), c_coeff in c_block_coeffs.items():
                for (sk, si), sub_coeff in sub_c_coeffs.items():
                    # C1 index: (bk, bi) in the 4x4 part of C
                    # C[k*4 + i] where k = bk*2+sk, i = bi*2+si
                    idx = (bk * 2 + sk) * 4 + (bi * 2 + si)
                    val = c_coeff * sub_coeff
                    if val != 0:
                        w_indices.append(idx)
                        w_vals.append(val)
            
            add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    
    # Part 2: 4x4 * 4x3 direct decomposition (rank 48)
    # A is 4x4, B2 is 4x3 (columns 4,5,6 of B)
    # C2 is 4x3, C[k*4 + i] where k=4,5,6; i=0,1,2,3
    # C2[k,i] = sum_j A[i,j] * B2[j,k-4]
    # = sum_j A[i,j] * B[j*7 + (k-4)]
    
    for i in range(4):
        for j in range(4):
            for k in range(3):
                # A index: i*4 + j
                # B index: j*7 + (4+k)
                # C index: (4+k)*4 + i
                u_idx = i * 4 + j
                v_idx = j * 7 + (4 + k)
                w_idx = (4 + k) * 4 + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
