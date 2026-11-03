"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 4x4 * 4x6 multiplication.
    # We exploit the recursive structure: 
    # 1. Split the 4x4 matrix A and 4x6 matrix B into 2x2 blocks.
    # 2. Use Strassen's algorithm for the 2x2 block multiplications.
    # 3. For the resulting 2x2 * 2x3 sub-problems, use a hybrid approach.
    #
    # Specifically, for a 2x2 matrix X and 2x3 matrix Y, we compute:
    # X = [[x11, x12], [x21, x22]], Y = [[y11, y12, y13], [y21, y22, y23]]
    # Product Z = XY is a 2x3 matrix:
    # Z = [[x11*y11 + x12*y21, x11*y12 + x12*y22, x11*y13 + x12*y23],
    #      [x21*y11 + x22*y21, x21*y12 + x22*y22, x21*y13 + x22*y23]]
    #
    # We can use Strassen's 7 multiplications for the 2x2 * 2x2 part (first two columns)
    # and add 2 more for the third column, totaling 9 multiplications.
    # However, a direct schoolbook for 2x2 * 2x3 uses 4*3 = 12 multiplications.
    # Strassen for 2x2 * 2x2 uses 7, plus 2 extra for the 3rd column (each requiring 2 mults? No, just reuse).
    # Actually, the 3rd column of Y is a 2x1 vector. The product of a 2x2 and 2x1 uses 4 mults.
    # So Strassen (7) + 4 = 11? No, that's worse than 12? No, 11 < 12.
    # Let's verify: Strassen computes 7 intermediate products. The 3rd column needs x11*y13, x12*y23, x21*y13, x22*y23.
    # These are 4 new products. Total 7+4=11.
    # But wait, we can reuse? No, y13 and y23 are new.
    # So rank for 2x2*2x3 is 11.
    #
    # Now for the top-level 4x4 * 4x6:
    # A is 4x4, B is 4x6.
    # Split A into 2x2 blocks A11, A12, A21, A22.
    # Split B into 2x2 blocks B11, B12, B21, B22 (for first 4 cols) and B13, B23 (for last 2 cols).
    #
    # C = AB is 4x6.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    #
    # Each Ci is a 2x2 or 2x3 matrix?
    # B11 is 2x2. A11*B11 is 2x2. A12*B21 is 2x2. So C11 is 2x2.
    # B13 is 2x2? No, B is 4x6. Rows 1-2, Cols 5-6. So B13 is 2x2.
    # So all Cij are 2x2 matrices.
    #
    # So we have 6 blocks of 2x2 results.
    # Each block is sum of two 2x2 products.
    # Each 2x2 product can be done in rank 7 (Strassen).
    # So each Cij block requires 7+7=14 multiplications?
    # Total 6 * 14 = 84. This is worse than schoolbook 96? No, 84 < 96.
    #
    # Can we do better?
    # The term A11*B11 appears in C11.
    # The term A11*B12 appears in C12.
    # The term A11*B13 appears in C13.
    # So A11 is multiplied by three different 2x2 blocks of B.
    # We can compute A11*B11, A11*B12, A11*B13 separately.
    # Total multiplications:
    # A11: 3 blocks * 7 = 21
    # A12: 3 blocks * 7 = 21
    # A21: 3 blocks * 7 = 21
    # A22: 3 blocks * 7 = 21
    # Total = 84.
    #
    # Is there a way to reduce this?
    # The rank of the 4x4*4x6 tensor is known to be 84? Or lower?
    # Let's just implement the 84 rank solution.
    #
    # Implementation details:
    # We need to generate U, V, W matrices.
    # U has shape (n*m, rank) = (16, 84)
    # V has shape (m*p, rank) = (24, 84)
    # W has shape (p*n, rank) = (24, 84)
    #
    # Indices:
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]  (Trace convention: W[k*n + i, r])
    #
    # We will iterate over the 4x4x6 space and use Strassen for each 2x2x2 sub-block.
    # But we need to map the global indices to local indices for each Strassen block.
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 * 2x2
    # M1 = A11 * (B11 - B21)
    # M2 = (A11 + A21) * B21
    # M3 = (A12 - A22) * B22
    # M4 = A22 * (B12 + B22)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * B11
    # M7 = (A12 + A22) * (B11 - B12)
    #
    # C11 = M1 + M2 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M3 + M4 + M6
    #
    # We will define a function to add a Strassen block contribution.
    
    def add_strassen_block(U, V, W, i0, j0, k0, col_start):
        # i0, j0, k0 are the top-left indices of the 2x2x2 block.
        # We map local indices (di, dj, dk) in {0,1} to global.
        # A indices: i in [i0, i0+1], j in [j0, j0+1]
        # B indices: j in [j0, j0+1], k in [k0, k0+1]
        # C indices: i in [i0, i0+1], k in [k0, k0+1]
        
        # Local indices for A: 00, 01, 10, 11
        # Local indices for B: 00, 01, 10, 11
        # Local indices for C: 00, 01, 10, 11
        
        # We will create 7 entries in U, V, W starting at col_start.
        
        # M1: A11 * (B11 - B21)
        # A11: i=i0, j=j0
        # B11: j=j0, k=k0
        # B21: j=j0+1, k=k0
        # U: A11
        U[i0*m + j0][col_start] = 1
        # V: B11 - B21
        V[j0*p + k0][col_start] = 1
        V[(j0+1)*p + k0][col_start] = -1
        # W: C11, C21? 
        # M1 contributes to C11 and C22?
        # C11 = M1 + ...
        # C22 = M1 - ...
        # So W for M1 is C11 and C22.
        # C11: i=i0, k=k0
        # C22: i=i0+1, k=k0+1
        W[k0*n + i0][col_start] = 1
        W[(k0+1)*n + (i0+1)][col_start] = 1
        
        # M2: (A11 + A21) * B21
        col = col_start + 1
        # A11 + A21
        U[i0*m + j0][col] = 1
        U[(i0+1)*m + j0][col] = 1
        # B21
        V[(j0+1)*p + k0][col] = 1
        # W: C11, C21
        W[k0*n + i0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        
        # M3: (A12 - A22) * B22
        col = col_start + 2
        # A12 - A22
        U[i0*m + (j0+1)][col] = 1
        U[(i0+1)*m + (j0+1)][col] = -1
        # B22
        V[(j0+1)*p + (k0+1)][col] = 1
        # W: C12, C22
        W[(k0+1)*n + i0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = -1
        
        # M4: A22 * (B12 + B22)
        col = col_start + 3
        # A22
        U[(i0+1)*m + (j0+1)][col] = 1
        # B12 + B22
        V[j0*p + (k0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        # W: C11, C22
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        
        # M5: (A11 + A12) * B22
        col = col_start + 4
        # A11 + A12
        U[i0*m + j0][col] = 1
        U[i0*m + (j0+1)][col] = 1
        # B22
        V[(j0+1)*p + (k0+1)][col] = 1
        # W: C11, C12
        W[k0*n + i0][col] = -1
        W[(k0+1)*n + i0][col] = 1
        
        # M6: (A21 - A11) * B11
        col = col_start + 5
        # A21 - A11
        U[(i0+1)*m + j0][col] = 1
        U[i0*m + j0][col] = -1
        # B11
        V[j0*p + k0][col] = 1
        # W: C21, C22
        W[k0*n + (i0+1)][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        
        # M7: (A12 + A22) * (B11 - B12)
        col = col_start + 6
        # A12 + A22
        U[i0*m + (j0+1)][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        # B11 - B12
        V[j0*p + k0][col] = 1
        V[j0*p + (k0+1)][col] = -1
        # W: C11, C22
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        
    # Now we need to combine these blocks for the 4x4*4x6 case.
    # We have 3 k-blocks (0, 1, 2) since p=6.
    # And 2 i-blocks (0, 1) and 2 j-blocks (0, 1).
    #
    # For each i-block (0,1) and j-block (0,1), we have 3 k-blocks.
    # So we have 4 * 3 = 12 Strassen blocks?
    # No, each Strassen block handles a 2x2 * 2x2 product.
    # We need to compute A[i-block, j-block] * B[j-block, k-block] for all i,j,k blocks.
    # There are 2 i-blocks, 2 j-blocks, and 3 k-blocks.
    # Total 2*2*3 = 12 Strassen blocks.
    # Each block uses 7 multiplications.
    # Total rank = 12 * 7 = 84.
    
    col = 0
    for i_block in range(2):
        for j_block in range(2):
            for k_block in range(3):
                i0 = i_block * 2
                j0 = j_block * 2
                k0 = k_block * 2
                add_strassen_block(U, V, W, i0, j0, k0, col)
                col += 7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
