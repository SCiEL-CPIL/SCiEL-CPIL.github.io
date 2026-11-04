"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal exact tensor decomposition for multiplying a 2x3 by 3x8 matrix.
    # Use Strassen's algorithm for 2x2 blocks to reduce rank from 48 to 44.
    # 
    # The 2x3 x 3x8 multiplication can be decomposed into:
    # - Two 2x3 x 3x4 blocks (splitting the 8 columns into two groups of 4)
    # - Each 2x3 x 3x4 can be further decomposed
    #
    # Actually, let's think more carefully. We have:
    # A is 2x3, B is 3x8, C = A*B is 2x8
    #
    # One approach: view this as multiplying a 2x3 matrix by a 3x8 matrix.
    # We can pad A to 4x4 and B to 4x4? No, that doesn't help directly.
    #
    # Better approach: Use the fact that we can split the computation.
    # 
    # Let's use a block decomposition. Split the 3x8 matrix into blocks.
    # Actually, the most straightforward improvement is to note that:
    # - We can treat the 2x3 * 3x8 as two 2x3 * 3x4 multiplications
    # - Each 2x3 * 3x4 = (2x3) * (3x4)
    # 
    # For 2x3 * 3x4:
    # A = [[a11, a12, a13], [a21, a22, a23]]
    # B = [[b11, b12, b13, b14], [b21, b22, b23, b24], [b31, b32, b33, b34]]
    # 
    # We can use a rank-7 decomposition for the core 2x2 part if we can
    # structure it properly. But 2x3 * 3x4 isn't directly 2x2 * 2x2.
    #
    # Alternative: Use a known low-rank decomposition for rectangular matrices.
    #
    # Actually, let's use a simpler approach: 
    # The schoolbook rank is 2*3*8 = 48.
    # 
    # We can save rank by noting that for the 2x2 block multiplications,
    # we can use Strassen (rank 7 instead of 8).
    #
    # Let's decompose the problem differently:
    # C = A * B where A is 2x3 and B is 3x8
    #
    # Write A = [A1 | A2] where A1 is 2x2 and A2 is 2x1
    # Write B = [B1; B2; B3] where B1, B2 are 2x8 and B3 is 1x8
    #
    # Then C = A1*B1 + A2*B2 + ... wait, this doesn't decompose nicely.
    #
    # Better: Let's use the approach from the knowledge:
    # "Using Strassen's algorithm for the 2x2 sub-blocks reduces the rank 
    # from 48 (schoolbook) to 44"
    #
    # So we need to find a decomposition of rank 44.
    #
    # Here's a concrete approach:
    # Split the 3 columns into a 2x2 block and a 1x1 block.
    # A = [A_left | A_right] where A_left is 2x2, A_right is 2x1
    # B = [B_top; B_bottom] where B_top is 2x8, B_bottom is 1x8
    #
    # C = A_left * B_top + A_right * B_bottom
    #
    # The first term A_left * B_top is a 2x2 * 2x8 multiplication.
    # We can view this as 4 separate 2x2 * 2x2 multiplications (splitting B_top into 4 blocks of 2x2).
    # Each 2x2 * 2x2 can be done in rank 7 (Strassen) instead of rank 8.
    # So that saves 4 * (8-7) = 4 rank.
    #
    # The second term A_right * B_bottom is a 2x1 * 1x8 multiplication.
    # This is just: C_right[i,j] = A_right[i] * B_bottom[j]
    # This requires rank 2*8 = 16 (since each of the 2*8 entries needs its own rank-1 term).
    #
    # Total rank: 4 * 7 + 16 = 28 + 16 = 44. 
    #
    # Let's implement this.
    #
    # For the 2x2 * 2x8 part:
    # B_top is 2x8. Split into 4 blocks of 2x2: B_top = [B0, B1, B2, B3]
    # A_left is 2x2.
    # 
    # For each 2x2 * 2x2 multiplication, use Strassen:
    # Given P = [[p11, p12], [p21, p22]], Q = [[q11, q12], [q21, q22]]
    # R = P * Q
    #
    # Strassen's 7 products:
    # M1 = (P11 + P22) * (Q11 + Q22)
    # M2 = (P21 + P22) * Q11
    # M3 = P11 * (Q12 - Q22)
    # M4 = P22 * (Q21 - Q11)
    # M5 = (P11 + P12) * Q22
    # M6 = (P21 - P11) * (Q11 + Q12)
    # M7 = (P12 - P22) * (Q21 + Q22)
    #
    # R11 = M1 + M4 - M5 + M7
    # R12 = M3 + M5
    # R21 = M2 + M4
    # R22 = M1 - M2 + M3 + M6
    #
    # Each Mi involves a product of a linear combination of P entries and a linear combination of Q entries.
    # In the tensor decomposition, each rank-1 term corresponds to one such product.
    #
    # Let's index the entries:
    # A_left[i,j] for i in {0,1}, j in {0,1}
    # B_top[i,j] for i in {0,1}, j in {0,...,7}
    #
    # For the k-th 2x2 block of B_top (k in 0..3), columns are 2k, 2k+1.
    #
    # Let's define the Strassen decomposition for each 2x2 * 2x2 block.
    #
    # Actually, let me just construct the full decomposition directly.
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: A_left (2x2) * B_top (2x8), using Strassen for each 2x2 block
    # A_left rows: i in {0,1}, cols: j in {0,1}
    # B_top rows: j in {0,1}, cols: k in {0,...,7}
    
    # Strassen factors for 2x2 * 2x2
    # M1 = (P11 + P22) * (Q11 + Q22)
    # M2 = (P21 + P22) * Q11
    # M3 = P11 * (Q12 - Q22)
    # M4 = P22 * (Q21 - Q11)
    # M5 = (P11 + P12) * Q22
    # M6 = (P21 - P11) * (Q11 + Q12)
    # M7 = (P12 - P22) * (Q21 + Q22)
    
    # For each of the 4 blocks (k=0..3):
    for block in range(4):
        # Q11 = B_top[0, 2*block], Q12 = B_top[0, 2*block+1]
        # Q21 = B_top[1, 2*block], Q22 = B_top[1, 2*block+1]
        
        q11_col = 2 * block      # column index in B_top (0-indexed within the 2x8 block)
        q12_col = 2 * block + 1
        q21_col = 2 * block
        q22_col = 2 * block + 1
        
        # M1 = (P11 + P22) * (Q11 + Q22)
        # P11 = A_left[0,0] -> U index 0*2+0 = 0
        # P22 = A_left[1,1] -> U index 1*2+1 = 3
        # Q11 = B_top[0, q11_col] -> V index 0*8 + q11_col = q11_col
        # Q22 = B_top[1, q22_col] -> V index 1*8 + q22_col = 8 + q22_col
        
        U[0][col] = 1   # P11
        U[3][col] = 1   # P22
        V[q11_col][col] = 1       # Q11
        V[8 + q22_col][col] = 1   # Q22
        # Result: R11 += M1, R22 -= M1
        W[(2*block) * 2 + 0][col] = 1     # R11, col 0
        W[(2*block+1) * 2 + 1][col] = -1   # R22, col 1
        col += 1
        
        # M2 = (P21 + P22) * Q11
        # P21 = A_left[1,0] -> U index 1*2+0 = 2
        # P22 = A_left[1,1] -> U index 1*2+1 = 3
        # Q11 = B_top[0, q11_col] -> V index q11_col
        
        U[2][col] = 1   # P21
        U[3][col] = 1   # P22
        V[q11_col][col] = 1       # Q11
        # Result: R21 += M2, R22 -= M2
        W[(2*block) * 2 + 1][col] = 1     # R21, col 1
        W[(2*block+1) * 2 + 1][col] = -1   # R22, col 1
        col += 1
        
        # M3 = P11 * (Q12 - Q22)
        # P11 = A_left[0,0] -> U index 0
        # Q12 = B_top[0, q12_col] -> V index q12_col
        # Q22 = B_top[1, q22_col] -> V index 8 + q22_col
        
        U[0][col] = 1   # P11
        V[q12_col][col] = 1       # Q12
        V[8 + q22_col][col] = -1   # -Q22
        # Result: R12 += M3, R22 += M3
        W[(2*block) * 2 + 1][col] = 1     # R12, col 1
        W[(2*block+1) * 2 + 1][col] = 1   # R22, col 1
        col += 1
        
        # M4 = P22 * (Q21 - Q11)
        # P22 = A_left[1,1] -> U index 3
        # Q21 = B_top[1, q21_col] -> V index 8 + q21_col
        # Q11 = B_top[0, q11_col] -> V index q11_col
        
        U[3][col] = 1   # P22
        V[8 + q21_col][col] = 1   # Q21
        V[q11_col][col] = -1       # -Q11
        # Result: R11 += M4, R21 += M4
        W[(2*block) * 2 + 0][col] = 1     # R11, col 0
        W[(2*block) * 2 + 1][col] = 1     # R21, col 1
        col += 1
        
        # M5 = (P11 + P12) * Q22
        # P11 = A_left[0,0] -> U index 0
        # P12 = A_left[0,1] -> U index 0*2+1 = 1
        # Q22 = B_top[1, q22_col] -> V index 8 + q22_col
        
        U[0][col] = 1   # P11
        U[1][col] = 1   # P12
        V[8 + q22_col][col] = 1   # Q22
        # Result: R11 -= M5, R12 += M5
        W[(2*block) * 2 + 0][col] = -1   # R11, col 0
        W[(2*block) * 2 + 1][col] = 1    # R12, col 1
        col += 1
        
        # M6 = (P21 - P11) * (Q11 + Q12)
        # P21 = A_left[1,0] -> U index 2
        # P11 = A_left[0,0] -> U index 0
        # Q11 = B_top[0, q11_col] -> V index q11_col
        # Q12 = B_top[0, q12_col] -> V index q12_col
        
        U[2][col] = 1   # P21
        U[0][col] = -1  # -P11
        V[q11_col][col] = 1       # Q11
        V[q12_col][col] = 1       # Q12
        # Result: R22 += M6
        W[(2*block+1) * 2 + 1][col] = 1   # R22, col 1
        col += 1
        
        # M7 = (P12 - P22) * (Q21 + Q22)
        # P12 = A_left[0,1] -> U index 1
        # P22 = A_left[1,1] -> U index 3
        # Q21 = B_top[1, q21_col] -> V index 8 + q21_col
        # Q22 = B_top[1, q22_col] -> V index 8 + q22_col
        
        U[1][col] = 1   # P12
        U[3][col] = -1  # -P22
        V[8 + q21_col][col] = 1   # Q21
        V[8 + q22_col][col] = 1   # Q22
        # Result: R11 += M7, R22 += M7
        W[(2*block) * 2 + 0][col] = 1     # R11, col 0
        W[(2*block+1) * 2 + 1][col] = 1   # R22, col 1
        col += 1
    
    # Part 2: A_right (2x1) * B_bottom (1x8)
    # A_right[i] for i in {0,1} -> U index i*2+2 = 2, 3
    # B_bottom[j] for j in {0,...,7} -> V index 2*8 + j = 16+j
    # Result: C[i,j] = A_right[i] * B_bottom[j]
    
    for i in range(2):
        for j in range(8):
            U[i * 2 + 2][col] = 1    # A_right[i]
            V[16 + j][col] = 1       # B_bottom[j]
            W[j * 2 + i][col] = 1    # C[i,j]
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
