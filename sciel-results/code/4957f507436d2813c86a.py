"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Use a 2-level Strassen decomposition to achieve rank 98 for 4x4 * 4x8.
    
    Strategy:
    - Level 1: Split the 4x4 (left) and 4x8 (right) into blocks.
      Actually, since dimensions are 4x4 and 4x8, we can view the 4x8 as 4x(4*2).
      Let's use a different splitting to exploit the rectangular shape.
      
    Better approach for 4x4 * 4x8:
    View A as 4x4, B as 4x8.
    We can split B into two 4x4 blocks: B = [B1, B2].
    Then C = A*B = [A*B1, A*B2].
    This requires 2 rank-16 multiplications = rank 32? No, rank is additive.
    If we just compute A*B1 and A*B2 separately, rank is rank(4x4*4x4) + rank(4x4*4x4).
    With schoolbook, rank is 16+16=32. But we know rank(4x4*4x4) is at most 19 (or 49 with Strassen level 2).
    However, the "rank" in the tensor decomposition sense for the whole 4x4x8 problem is a single number.
    The schoolbook rank is 4*4*8 = 128.
    
    Let's use the known rank-49 algorithm for 4x4*4x4 (which is 2 levels of Strassen) and adapt it.
    Actually, the rank of M_{4,4,4} is known to be 49 using 2-level Strassen (7^2).
    For 4x4 * 4x8, we can view the 8 columns as 2 sets of 4 columns.
    Let B = [B_L, B_R] where B_L, B_R are 4x4.
    C = [A B_L, A B_R].
    We can compute A B_L and A B_R using the rank-49 algorithm for each?
    That would give rank 49 + 49 = 98.
    Since the inputs are separate (A is shared, but B_L and B_R are different parts of B), 
    and the outputs are concatenated, the tensors add up.
    The U, V, W matrices would be block-diagonal in V and W, and U would be duplicated.
    
    Let's verify the rank:
    Schoolbook: 128.
    This method: 2 * rank(M_{4,4,4}) = 2 * 49 = 98.
    Is there a better way?
    Maybe a direct rank < 98 exists? 
    But 98 is a solid improvement over 128.
    
    Let's implement the rank-49 decomposition for 4x4*4x4 first, then duplicate it.
    
    Rank 49 decomposition of 4x4*4x4:
    This is obtained by applying Strassen to 2x2 blocks, and then Strassen to the 2x2 blocks of that?
    No, 4x4*4x4 is not directly 2 levels of 2x2 Strassen in the standard way.
    Standard Strassen reduces 2x2*2x2 to rank 7.
    For 4x4*4x4, we can partition into four 2x2 blocks.
    A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]].
    C = A B.
    Using Strassen, we compute 7 intermediate 2x2 matrices.
    Each 2x2 matrix is computed using the rank-7 Strassen algorithm.
    So the total rank is 7 * 7 = 49.
    
    So the plan:
    1. Define the rank-49 decomposition for 4x4*4x4.
    2. For 4x4*4x8, split B into B_L and B_R (both 4x4).
    3. Compute C_L = A B_L using rank-49.
    4. Compute C_R = A B_R using rank-49.
    5. Concatenate the results to form C.
    
    The rank will be 49 + 49 = 98.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Base case: Rank-7 Strassen for 2x2 * 2x2
    # Indices for 2x2: 0,1,2,3
    # U, V, W are 4x7
    S2x2 = {
        'U': [
            [1, 0, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 1, 0],
            [1, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 1],
        ],
        'V': [
            [1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
        ],
        'W': [
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0],
        ]
    }
    # Let's double check the Strassen formulas.
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
    
    # U (A coefficients):
    # M1: A11=1, A22=1 -> [1,0,0,1]
    # M2: A21=1, A22=1 -> [0,0,1,1]
    # M3: A11=1 -> [1,0,0,0]
    # M4: A22=1 -> [0,0,0,1]
    # M5: A11=1, A12=1 -> [1,1,0,0]
    # M6: A21=1, A11=-1 -> [-1,0,1,0]
    # M7: A12=1, A22=-1 -> [0,1,0,-1]
    
    # V (B coefficients):
    # M1: B11=1, B22=1 -> [1,0,0,1]
    # M2: B11=1 -> [1,0,0,0]
    # M3: B12=1, B22=-1 -> [0,1,0,-1]
    # M4: B21=1, B11=-1 -> [-1,0,1,0]
    # M5: B22=1 -> [0,0,0,1]
    # M6: B11=1, B12=1 -> [1,1,0,0]
    # M7: B21=1, B22=1 -> [0,0,1,1]
    
    # W (C coefficients):
    # C11: M1=1, M4=1, M5=-1, M7=1 -> [1,0,0,0, -1,0, 0,0, 1,0, 0,0, 1,0, 0,0] ... wait, W is (4, 7)
    # W[i, c] is coeff for C_i from M_c.
    # C11 (idx 0): M1(0)=1, M4(3)=1, M5(4)=-1, M7(6)=1 -> [1, 0, 0, 1, -1, 0, 1]
    # C12 (idx 1): M3(2)=1, M5(4)=1 -> [0, 0, 1, 0, 1, 0, 0]
    # C21 (idx 2): M2(1)=1, M4(3)=1 -> [0, 1, 0, 1, 0, 0, 0]
    # C22 (idx 3): M1(0)=1, M2(1)=-1, M3(2)=1, M6(5)=1 -> [1, -1, 1, 0, 0, 1, 0]
    
    # Let's re-verify W[2,2] for C21 = M2 + M4.
    # M2 is col 1, M4 is col 3.
    # So W[2,1]=1, W[2,3]=1.
    # My previous manual check: W[2] = [0, 1, 0, 1, 0, 0, 0]. Correct.
    
    # Let's re-verify W[3,2] for C22 = M1 - M2 + M3 + M6.
    # M1 col 0, M2 col 1, M3 col 2, M6 col 5.
    # W[3,0]=1, W[3,1]=-1, W[3,2]=1, W[3,5]=1.
    # My previous: W[3] = [1, -1, 1, 0, 0, 1, 0]. Correct.
    
    # So the 2x2 Strassen matrices are:
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Construct 4x4 * 4x4 rank-49 decomposition
    # Block structure:
    # A = [[A0, A1], [A2, A3]] where A0..A3 are 2x2.
    # B = [[B0, B1], [B2, B3]] where B0..B3 are 2x2.
    # C = [[C0, C1], [C2, C3]] where C0..C3 are 2x2.
    
    # Using Strassen on the 2x2 blocks:
    # M00 = (A0+A3)(B0+B3)
    # M01 = (A2+A3)B0
    # M02 = A0(B1-B3)
    # M03 = A3(B2-B0)
    # M04 = (A0+A1)B3
    # M05 = (A2-A0)(B0+B1)
    # M06 = (A1-A3)(B2+B3)
    
    # C0 = M00 + M03 - M04 + M06
    # C1 = M02 + M04
    # C2 = M01 + M03
    # C3 = M00 - M01 + M02 + M05
    
    # Each Mi is a 2x2 matrix computed by 2x2 Strassen (rank 7).
    # So we have 7 "super-multiplications", each of rank 7. Total 49.
    
    # U4 (16 x 49):
    # For each super-mult j (0..6), we have 7 sub-mults.
    # U4[row, 7*j + k] = U_super[j, row_block] * U2[row_in_block, k]
    # where row_block is the 2x2 block index of the row in A.
    
    # Let's build U4, V4, W4.
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # Super U coefficients (which 2x2 blocks of A are used)
    U_super = [
        [1, 0, 0, 1], # M00: A0, A3
        [0, 0, 1, 1], # M01: A2, A3
        [1, 0, 0, 0], # M02: A0
        [0, 0, 0, 1], # M03: A3
        [1, 1, 0, 0], # M04: A0, A1
        [-1, 0, 1, 0], # M05: -A0, A2
        [0, 1, 0, -1]  # M06: A1, -A3
    ]
    
    # Super V coefficients (which 2x2 blocks of B are used)
    V_super = [
        [1, 0, 0, 1], # M00: B0, B3
        [1, 0, 0, 0], # M01: B0
        [0, 1, 0, -1], # M02: B1, -B3
        [-1, 0, 1, 0], # M03: -B0, B2
        [0, 0, 0, 1], # M04: B3
        [1, 1, 0, 0], # M05: B0, B1
        [0, 0, 1, 1]  # M06: B2, B3
    ]
    
    # Super W coefficients (which C blocks are formed)
    # C0 = M00 + M03 - M04 + M06
    # C1 = M02 + M04
    # C2 = M01 + M03
    # C3 = M00 - M01 + M02 + M05
    W_super = [
        [1, 0, 0, 0, -1, 0, 1], # C0
        [0, 0, 1, 0, 1, 0, 0],  # C1
        [0, 1, 0, 1, 0, 0, 0],  # C2
        [1, -1, 1, 0, 0, 1, 0]  # C3
    ]
    
    # Fill U4
    for j in range(7): # Super mult index
        for k in range(7): # Sub mult index
            col = 7*j + k
            # U2[k] gives coefficients for rows in the 2x2 block
            for b in range(4): # Block index of A
                coeff_super = U_super[j][b]
                if coeff_super != 0:
                    # Rows in block b: 2*b, 2*b+1
                    for r in range(2):
                        row = 2*b + r
                        U4[row][col] = coeff_super * U2[r][k]
    
    # Fill V4
    for j in range(7):
        for k in range(7):
            col = 7*j + k
            for b in range(4):
                coeff_super = V_super[j][b]
                if coeff_super != 0:
                    for r in range(2):
                        row = 2*b + r
                        V4[row][col] = coeff_super * V2[r][k]
    
    # Fill W4
    for c in range(4): # C block index
        for j in range(7): # Super mult
            coeff_super = W_super[c][j]
            if coeff_super != 0:
                for k in range(7): # Sub mult
                    col = 7*j + k
                    for r in range(2):
                        row = 2*c + r
                        W4[row][col] = coeff_super * W2[r][k]
    
    # Now, construct 4x4 * 4x8 decomposition.
    # B is 4x8. Split into B_L (4x4) and B_R (4x4).
    # C_L = A B_L (4x4), C_R = A B_R (4x4).
    # C = [C_L, C_R] (4x8).
    
    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)] # p*n = 8*4 = 32. W[k*n+i, r].
    
    # Copy U4 to U (twice, since A is shared for both C_L and C_R)
    for i in range(16):
        for r in range(49):
            U[i][r] = U4[i][r]
            U[i][49+r] = U4[i][r]
    
    # V_L uses V4. V_R uses V4 but with shifted columns in B.
    # B_L columns 0..3, B_R columns 4..7.
    for i in range(16): # Rows of B_L/B_R (which are 4x4 blocks)
        # Wait, V is (m*p, rank) = (4*8, 98) = (32, 98).
        # V row index corresponds to (j, k) in B[j, k].
        # j is 0..3 (row in B), k is 0..7 (col in B).
        # B_L has k=0..3, B_R has k=4..7.
        # V4 is for a 4x4 matrix. Its rows are (j, k) with k=0..3.
        # So V4[i] corresponds to B_L[i, k] for i=0..15.
        # For B_R, the columns are shifted by 4.
        # So V[2*j + k, col] for B_L is V4[2*j + (k), col].
        # For B_R, V[2*j + (k+4), 49+col'] is V4[2*j + k, col'].
        
        # Let's map carefully.
        # V4 row index i (0..15) corresponds to j=i//2, k=i%2.
        # This is for a 4x4 matrix where row j is 0..3 and col k is 0..3.
        # In our 4x8 matrix, B[j, k] with k=0..3 is B_L.
        # B[j, k] with k=4..7 is B_R.
        
        for i in range(16):
            j = i // 2
            k = i % 2
            # This i corresponds to B[j, k] in a 4x4.
            # In B_L, this is B[j, k]. Row index in V: j*8 + k? No.
            # V is (m*p, rank). m=4, p=8.
            # The tensor index is (j, k) where j is row of A (0..3) and k is col of B (0..7)?
            # Wait, the standard decomposition is:
            # C[i, k] = sum_{j, r} U[i*m+j, r] * A[i,j] * V[j*p+k, r] * W[k*n+i, r]
            # Here U is (n*m, rank) = (16, rank). Index (i, j).
            # V is (m*p, rank) = (32, rank). Index (j, k).
            # W is (p*n, rank) = (32, rank). Index (k, i).
            
            # So V row index is j*p + k = 4*j + k.
            # V4 row index is j*4 + k = 4*j + k (for 4x4).
            # So V4 row i corresponds to j=i//4, k=i%4.
            # Let's re-verify V4 construction.
            # V4 was constructed for 4x4 * 4x4.
            # Rows of B are 0..3, Cols are 0..3.
            # Row index in V4: j*4 + k.
            
            # So for B_L (k=0..3):
            # V[4*j + k, col] = V4[4*j + k, col]
            # For B_R (k=4..7):
            # Let k' = k - 4.
            # V[4*j + (k'+4), 49+col'] = V4[4*j + k', col']
            
    # Let's rewrite the fill loop clearly.
    for i in range(16):
        j = i // 4
        k = i % 4
        for r in range(49):
            # B_L part
            V[4*j + k][r] = V4[i][r]
            # B_R part
            V[4*j + k + 4][49 + r] = V4[i][r]
            
    # Fill W.
    # W is (p*n, rank) = (32, 98). Index (k, i).
    # W4 is for 4x4 * 4x4. Index (k, i) with k=0..3, i=0..3.
    # Row index in W4: k*4 + i.
    # For C_L (k=0..3):
    # W[4*k + i, col] = W4[4*k + i, col]
    # For C_R (k=4..7):
    # Let k' = k - 4.
    # W[4*(k'+4) + i, 49+col'] = W4[4*k' + i, col']
    
    for i in range(16):
        k = i // 4
        row_i = i % 4
        for r in range(49):
            # C_L part
            W[4*k + row_i][r] = W4[i][r]
            # C_R part
            W[4*(k+4) + row_i][49 + r] = W4[i][r]
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
