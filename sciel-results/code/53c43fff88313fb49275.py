"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-28 Strassen-based decomposition for 2x4x4 matrix multiplication.
    # This exploits the 4x4 multiplication (rank 28) with the 2x4 reduction.
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0.0] * rank for _ in range(p * n)]  # 8 x 28

    # Strassen decomposition for 4x4 multiplication.
    # The 4x4 multiplication can be decomposed into 7 multiplications of 2x2 blocks,
    # but we use the known rank-28 decomposition.
    # 
    # For 4x4, the standard Strassen-like decomposition uses 7 products of 2x2 matrices.
    # Each 2x2 multiplication has rank 4 (schoolbook) or we can use a more efficient scheme.
    # 
    # Actually, the bilinear complexity of 4x4 matrix multiplication is 28.
    # This is achieved by decomposing 4x4 into 7 blocks of 2x2, where each 2x2
    # multiplication uses a rank-4 scheme, but with shared terms.
    #
    # Let me use the explicit rank-28 decomposition for 4x4x4.
    # The indices for 4x4 multiplication: C[a][b] = sum_{c} A[a][c] * B[c][b]
    # where a, b, c in {0,1,2,3}
    #
    # The rank-28 decomposition of the 4x4 matrix multiplication tensor:
    # We partition the 4x4 matrices into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]], C = [[C00, C01], [C10, C11]]
    # Each block is 2x2.
    #
    # Using Strassen for 2x2: M1 = (A00+A11)*(B00+B11), M2 = (A10+A11)*B00,
    # M3 = A00*(B01-B11), M4 = A11*(B10-B00), M5 = (A00+A01)*B11,
    # M6 = (A10-A00)*(B00+B10), M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
    #
    # Each Mi is a 2x2 product, so we need to decompose each 2x2 product.
    # For a 2x2 product, the schoolbook has rank 4.
    # But we can't just use 7*4=28 if they share terms... actually 7*4=28 is exactly 28.
    #
    # Wait, that's the key. If we use Strassen's 7 products for 4x4, and each 2x2
    # product uses the schoolbook rank 4, we get 7*4 = 28 terms.
    #
    # Let me construct this.
    #
    # For the 2x4x4 problem:
    # A is 2x4, B is 4x4, C is 2x4.
    # C[i][j] = sum_{k} A[i][k] * B[k][j], i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    #
    # We can think of this as two rows of A (each 1x4) times B (4x4).
    # Or we can group: treat A as a 2x4 matrix.
    #
    # Alternative: embed into 4x4x4 by padding A to 4x4 (with zeros in rows 2,3),
    # then C (padded to 4x4) = A_padded * B. The upper-left 2x4 of C_padded is our C.
    #
    # But the rank-28 decomposition of 4x4x4 applies to the full 4x4 multiplication.
    # We only need the top 2 rows of the result.
    #
    # Let me directly construct the decomposition.
    #
    # For C[i][j] = sum_k A[i][k] B[k][j], we need bilinear forms.
    # 
    # Let's use the 4x4 Strassen decomposition adapted for 2x4x4.
    # Partition B into 2x2 blocks: B = [[B00, B01], [B10, B11]] where each is 2x2.
    # A has rows: A0 = A[0] (1x4), A1 = A[1] (1x4).
    # A0 = [A00, A01] where A00 = A[0][0:2], A01 = A[0][2:4] (each 1x2)
    # A1 = [A10, A11] where A10 = A[1][0:2], A11 = A[1][2:4] (each 1x2)
    #
    # C0 = A0 * B (1x4) = [A00*B00 + A01*B10, A00*B01 + A01*B11]
    # C1 = A1 * B (1x4) = [A10*B00 + A11*B10, A10*B01 + A11*B11]
    #
    # So C0 is 1x4, C1 is 1x4, total C is 2x4.
    #
    # Now, each product like A00*B00 is a (1x2)*(2x2) = 1x2 product.
    # A (1x2) times B (2x2): result[0] = a0*b00 + a1*b10, result[1] = a0*b01 + a1*b11
    # This has rank 2 (two terms).
    #
    # So C0[0] = A00*B00[0] + A01*B10[0]
    # C0[1] = A00*B00[1] + A01*B10[1]
    # etc.
    #
    # Total terms: 4 block products (A00*B00, A01*B10, A00*B01, A01*B11 for C0)
    # and 4 for C1. Each block product has rank 2. Total = 8 * 2 = 16 terms.
    # Wait, that's rank 16, which is better than 28!
    #
    # Let me recount. C is 2x4, so 8 elements.
    # C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0] + A[0][3]*B[3][0]
    # That's 4 terms for each element, 8 elements, so 32 terms (schoolbook).
    #
    # The block approach:
    # C0 (row 0, 4 elements) = A00*B00 + A01*B10
    # where A00 is 1x2, B00 is 2x2. Product is 1x2.
    # A01 is 1x2, B10 is 2x2. Product is 1x2.
    # So C0[0:2] = (A00*B00)[0:2] + (A01*B10)[0:2]
    # C0[2:4] = (A00*B01)[0:2] + (A01*B11)[0:2]
    #
    # Similarly for C1.
    #
    # Each (1x2)*(2x2) product:
    # Let a = [a0, a1], B = [[b00,b01],[b10,b11]]
    # result[0] = a0*b00 + a1*b10
    # result[1] = a0*b01 + a1*b11
    # Rank 2.
    #
    # So we have 8 such products (4 for C0, 4 for C1), each of rank 2.
    # Total rank = 8 * 2 = 16.
    #
    # Wait, can we do better? The 1x2 * 2x2 product is just a linear combination.
    # Actually, this is already quite good. Let me verify: 16 < 28 < 32.
    #
    # Let me construct this rank-16 decomposition.
    
    # U: 8 x 16, V: 16 x 16, W: 8 x 16
    # Wait, rank is 16, so U is 8x16, V is 16x16, W is 8x16.
    
    rank = 16
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 16
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 16
    W = [[0.0] * rank for _ in range(p * n)]  # 8 x 16
    
    # Indices:
    # U row index: i*m + j = i*4 + j, i in {0,1}, j in {0,1,2,3}
    # V row index: j*p + k = j*4 + k, j in {0,1,2,3}, k in {0,1,2,3}
    # W row index: k*n + i = k*2 + i, k in {0,1,2,3}, i in {0,1}
    
    # 8 block products:
    # For C row i (i=0,1), B block column (bc=0,1), B block row (br=0,1):
    # A_block = A[i][br*2 : br*2+2]  (1x2)
    # B_block = B[br*2 : br*2+2][bc*2 : bc*2+2]  (2x2)
    # Product is 1x2, contributing to C[i][bc*2 : bc*2+2]
    
    # Each product has 2 terms.
    # Term 1: A[i][br*2] * B[br*2][bc*2] + A[i][br*2+1] * B[br*2+1][bc*2] -> C[i][bc*2]
    # Term 2: A[i][br*2] * B[br*2][bc*2+1] + A[i][br*2+1] * B[br*2+1][bc*2+1] -> C[i][bc*2+1]
    
    col = 0
    for i in range(2):  # C row
        for br in range(2):  # B block row (and A sub-column block)
            for bc in range(2):  # B block column (C sub-column block)
                # Term 1: C[i][bc*2]
                # A[i][br*2] * B[br*2][bc*2]
                U[i * 4 + br * 2][col] = 1.0
                V[(br * 2) * 4 + (bc * 2)][col] = 1.0
                W[(bc * 2) * 2 + i][col] = 1.0
                # A[i][br*2+1] * B[br*2+1][bc*2]
                U[i * 4 + br * 2 + 1][col] = 1.0
                V[(br * 2 + 1) * 4 + (bc * 2)][col] = 1.0
                W[(bc * 2) * 2 + i][col] = 1.0
                col += 1
                
                # Term 2: C[i][bc*2+1]
                # A[i][br*2] * B[br*2][bc*2+1]
                U[i * 4 + br * 2][col] = 1.0
                V[(br * 2) * 4 + (bc * 2 + 1)][col] = 1.0
                W[(bc * 2 + 1) * 2 + i][col] = 1.0
                # A[i][br*2+1] * B[br*2+1][bc*2+1]
                U[i * 4 + br * 2 + 1][col] = 1.0
                V[(br * 2 + 1) * 4 + (bc * 2 + 1)][col] = 1.0
                W[(bc * 2 + 1) * 2 + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
