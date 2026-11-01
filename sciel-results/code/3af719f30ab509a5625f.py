"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-56 exact tensor decomposition for 2x2 * 2x16 multiplication.
    Uses Strassen's 7-term algorithm for each 2x2 block of the 2x16 matrix.
    """
    n, m, p = 2, 2, 16
    
    # Strassen's 7 multiplication formulas for 2x2 matrices:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each block b (0 or 1), we have 7 terms.
    # Total rank = 2 * 7 = 14? No wait.
    # The 2x2 matrix A is 2x2. The 2x16 matrix B is split into two 2x8 blocks? 
    # No, the problem is 2x2 * 2x16.
    # Let's view B as a 2x2 block matrix where each block is 2x8.
    # So B = [[B00, B01], [B10, B11]] where each Bxy is 2x8.
    # Then C = A * B = [[A11*B00 + A12*B10, A11*B01 + A12*B11],
    #                  [A21*B00 + A22*B10, A21*B01 + A22*B11]]
    # Each entry of C is a sum of two products of 2x2 * 2x8? No.
    # A is 2x2, B is 2x16.
    # Let's split B into two 2x8 blocks: B = [B_left, B_right].
    # C = [A*B_left, A*B_right].
    # Each A*B_block is 2x8.
    # We can use Strassen for 2x2 * 2x2, but here the second matrix is 2x8.
    # We can apply Strassen 4 times (for each 2x2 sub-block of the 2x8 matrix).
    # So for one 2x8 block, we need 4 * 7 = 28 terms.
    # For two 2x8 blocks, we need 2 * 28 = 56 terms.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    # Strassen coefficients
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indices for A (2x2):
    # A11: i=0,j=0 -> idx 0
    # A12: i=0,j=1 -> idx 1
    # A21: i=1,j=0 -> idx 2
    # A22: i=1,j=1 -> idx 3
    
    # For each 2x2 block of B (within a 2x8 block), the indices are:
    # B11, B12, B21, B22 corresponding to columns in that 2x2 block.
    # Let's define the Strassen terms for a 2x2 * 2x2 multiplication.
    # A is fixed. B varies.
    # For a given 2x2 sub-block of B, with columns c0, c1 (local indices),
    # and rows r0, r1 (local indices, which are 0,1 for the 2x16 matrix).
    
    # Let's define the 7 terms for a 2x2 * 2x2 multiplication.
    # Term 1: A=(A11+A22), B=(B11+B22)
    # Term 2: A=(A21+A22), B=B11
    # Term 3: A=A11, B=(B21-B22)
    # Term 4: A=A22, B=(B11-B21)
    # Term 5: A=(A11+A12), B=B22
    # Term 6: A=(A21-A11), B=(B11+B12)
    # Term 7: A=(A12-A22), B=(B21+B22)
    
    # Output reconstruction:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's iterate over the 4 blocks in B (each 2x2) and the 2 halves (left/right).
    # Actually, B is 2x16. We can view it as 8 blocks of 2x2.
    # Let's index the blocks by b=0..7.
    # Block b has columns 2*b, 2*b+1.
    # Rows are 0, 1.
    
    for b in range(8):
        # Columns for this block
        col0 = 2 * b
        col1 = 2 * b + 1
        
        # 7 terms for this block
        for t in range(7):
            col_idx = b * 7 + t
            if col_idx >= rank:
                break
                
            # Determine U, V, W for this term
            # U is 4x1, V is 4x1 (for the 2x2 block), W is 4x1 (for the 2x2 output block)
            # Wait, V is 32x56. The 2x2 block of B has 4 entries.
            # V[j*p+k] where j is row of B (0,1) and k is column of B (0..15).
            # So V indices for this block are:
            # B11: j=0, k=col0 -> idx 0*16 + col0 = col0
            # B12: j=0, k=col1 -> idx 0*16 + col1 = col1
            # B21: j=1, k=col0 -> idx 1*16 + col0 = 16+col0
            # B22: j=1, k=col1 -> idx 1*16 + col1 = 16+col1
            
            # U indices for A:
            # A11: 0, A12: 1, A21: 2, A22: 3
            
            # W indices for C (2x2 block):
            # C11: i=0, k=col0 -> idx 0*2 + col0? No.
            # W is (p*n) x rank = 32 x 56.
            # W[k*n+i] where k is column of C (0..15) and i is row of C (0,1).
            # So W index for C11 (i=0, k=col0) is col0*2 + 0 = 2*col0
            # C12 (i=0, k=col1) is col1*2 + 0 = 2*col1
            # C21 (i=1, k=col0) is col0*2 + 1 = 2*col0+1
            # C22 (i=1, k=col1) is col1*2 + 1 = 2*col1+1
            
            if t == 0:
                # M1 = (A11+A22)(B11+B22)
                U[0][col_idx] = 1  # A11
                U[3][col_idx] = 1  # A22
                V[col0][col_idx] = 1  # B11
                V[16+col1][col_idx] = 1  # B22
                # C11 = M1, C22 = M1
                W[2*col0][col_idx] = 1  # C11
                W[2*col1+1][col_idx] = 1  # C22
            elif t == 1:
                # M2 = (A21+A22)B11
                U[2][col_idx] = 1  # A21
                U[3][col_idx] = 1  # A22
                V[col0][col_idx] = 1  # B11
                # C21 = M2, C22 = -M2
                W[2*col0+1][col_idx] = 1  # C21
                W[2*col1+1][col_idx] = -1  # C22
            elif t == 2:
                # M3 = A11(B21-B22)
                U[0][col_idx] = 1  # A11
                V[16+col0][col_idx] = 1  # B21
                V[16+col1][col_idx] = -1  # B22
                # C12 = M3, C22 = M3
                W[2*col1][col_idx] = 1  # C12
                W[2*col1+1][col_idx] = 1  # C22
            elif t == 3:
                # M4 = A22(B11-B21)
                U[3][col_idx] = 1  # A22
                V[col0][col_idx] = 1  # B11
                V[16+col0][col_idx] = -1  # B21
                # C11 = M4, C21 = M4
                W[2*col0][col_idx] = 1  # C11
                W[2*col0+1][col_idx] = 1  # C21
            elif t == 4:
                # M5 = (A11+A12)B22
                U[0][col_idx] = 1  # A11
                U[1][col_idx] = 1  # A12
                V[16+col1][col_idx] = 1  # B22
                # C11 = -M5, C12 = M5
                W[2*col0][col_idx] = -1  # C11
                W[2*col1][col_idx] = 1  # C12
            elif t == 5:
                # M6 = (A21-A11)(B11+B12)
                U[2][col_idx] = 1  # A21
                U[0][col_idx] = -1  # A11
                V[col0][col_idx] = 1  # B11
                V[col1][col_idx] = 1  # B12
                # C22 = M6
                W[2*col1+1][col_idx] = 1  # C22
            elif t == 6:
                # M7 = (A12-A22)(B21+B22)
                U[1][col_idx] = 1  # A12
                U[3][col_idx] = -1  # A22
                V[16+col0][col_idx] = 1  # B21
                V[16+col1][col_idx] = 1  # B22
                # C11 = M7
                W[2*col0][col_idx] = 1  # C11
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
