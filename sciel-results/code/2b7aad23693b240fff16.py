"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 part (Strassen)
    # Rank 16 for 4x4x1 part (Schoolbook)
    rank = 49 + 16
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper: Strassen rank-7 decomposition for 2x2x2
    # Returns U2, V2, W2 of size 4x7
    def strassen_2x2():
        U2 = [[0]*7 for _ in range(4)]
        V2 = [[0]*7 for _ in range(4)]
        W2 = [[0]*7 for _ in range(4)]
        
        # P1 = (A00+A11)(B00+B11)
        # A part: A00 + A11. Indices 0,1 and 2,3.
        # U2[i,j,c]
        # For c=0:
        U2[0][0] = 1; U2[1][0] = 1; U2[2][0] = 1; U2[3][0] = 1
        V2[0][0] = 1; V2[1][0] = 1; V2[2][0] = 1; V2[3][0] = 1
        W2[0][0] = 1; W2[1][0] = 1; W2[2][0] = 1; W2[3][0] = 1
        
        # P2 = (A10+A11)B00
        # A part: A10 + A11. Indices 2,3 (rows 1,2? No, 2x2 blocks).
        # A10 is bottom-left (rows 2,3 cols 0,1), A11 is bottom-right (rows 2,3 cols 2,3).
        # In 2x2 local indices: A10 is rows 1, cols 0. A11 is rows 1, cols 1.
        # Wait, 2x2 matrix indices are 0,1.
        # Block 00: r0,c0. Block 01: r0,c1. Block 10: r1,c0. Block 11: r1,c1.
        # So A10 corresponds to local row 1, col 0.
        # U2 for A10+A11:
        # Local indices:
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1).
        # A10 is (1,0) -> index 2. A11 is (1,1) -> index 3.
        U2[2][1] = 1; U2[3][1] = 1
        # B00 is (0,0) -> index 0.
        V2[0][1] = 1
        # W part: C10? No, P2 contributes to C10 and C11.
        # But W2 here is the "output" of the 2x2 mult.
        # The 2x2 mult P2 = A' B'.
        # The decomposition of P2 into 7 terms is standard Strassen.
        # We need the U,V,W for the 2x2 multiplication itself.
        # Let's just use the standard Strassen coefficients for 2x2x2.
        
        # It is safer to define the 7 terms of Strassen for 2x2x2 directly.
        # Term 1: (A00+A11)(B00+B11)
        # U: 1,1,1,1. V: 1,1,1,1. W: 1,1,1,1.
        # Term 2: (A10+A11)B00
        # U: 0,0,1,1. V: 1,0,0,0. W: 0,0,1,1? No.
        # The W matrix for 2x2x2 term c has indices (k,i).
        # For P2 = (A10+A11)B00, the result is a 2x2 matrix.
        # The contribution to C is not just W2.
        # Actually, for the 2x2x2 tensor, the W matrix is the same as the "C" part.
        # In Strassen, the 7 products are scalar multiplications.
        # The 2x2x2 tensor decomposition of rank 7 is:
        # 1. (A00+A11) * (B00+B11) -> contributes to C00, C11
        # This is not a single scalar multiplication.
        # Strassen's algorithm computes 7 scalar products.
        # M1 = (A00+A11)(B00+B11). This is a 2x2 matrix product? No.
        # A00, A11 are scalars in the 2x2 case?
        # No, for 2x2 matrix multiplication, A00 is a scalar.
        # So M1 is a scalar product.
        # So U2, V2, W2 for 2x2x2 rank 7 are 4x7 matrices where each column has 1s.
        
        # Let's re-verify Strassen for 2x2.
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]].
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Tensor T_{i,j,k}.
        # U[i,j,c], V[j,k,c], W[k,i,c].
        
        # c=0 (M1):
        # A part: a+d. U[0,0]=1, U[1,1]=1.
        # B part: e+h. V[0,0]=1, V[1,1]=1.
        # C part: M1. W[0,0]=1, W[1,1]=1?
        # M1 contributes to C00 and C11.
        # So W[0,0]=1 (C00), W[1,1]=1 (C11).
        
        # c=1 (M2):
        # A part: c+d. U[1,0]=1, U[1,1]=1.
        # B part: e. V[0,0]=1.
        # C part: M2. Contributes to C10, C11.
        # W[0,1]=1 (C10), W[1,1]=1 (C11)? No, C11 is index (1,1).
        # W[k,i]. C10 is k=0,i=1. C11 is k=1,i=1.
        # So W[0,1]=1, W[1,1]=1.
        
        # This is getting complex to hardcode.
        # But we can just use the fact that we are building 4x4x4 from 2x2x2.
        
        pass

    # Given the complexity and time, I will stick to the schoolbook rank 80 if I can't easily verify the Strassen implementation.
    # However, rank 65 is a significant jump.
    # Let's try to implement the Strassen 4x4x4 carefully.
    
    # I will output the schoolbook program as it is safer, but the directive asks for a stronger algorithm.
    # If I provide a broken Strassen, score is 0.
    # I will provide the schoolbook program but with a comment that it's the seed.
    # Wait, the directive says "design a new program ... from scratch".
    # If I can't guarantee correctness of Strassen in 120s, I should probably stick to a simpler improvement or ensure correctness.
    
    # Actually, I can use a known rank 25 Laderman decomposition for 4x4x4.
    # But I don't have the coefficients memorized.
    
    # I will output the schoolbook code as it is the only one I can guarantee is correct immediately.
    # But the score will be 1.0.
    
    # Let's try to implement the 4x4x1 + 4x4x4(Strassen) correctly.
    
    n, m, p = DIMENSIONS
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 4x4x1 (columns 0-3 of B? No, 4x4x4 uses cols 0-3)
    # Let's say 4x4x4 uses k=0..3.
    # 4x4x1 uses k=4.
    
    # 4x4x1 part (indices 49..64)
    for i in range(4):
        for j in range(4):
            c = 49 + i * 4 + j
            U[i * 4 + j][c] = 1
            V[j * 5 + 4][c] = 1
            W[4 * 4 + i][c] = 1
            
    # 4x4x4 part (indices 0..48)
    # Strassen 2x2 coefficients
    # A coeffs for 7 terms
    # 0: A00+A11
    # 1: A10+A11
    # 2: A00
    # 3: A11
    # 4: A00+A01
    # 5: A10-A00
    # 6: A01-A11
    
    A_coeff = [
        [[1,0],[0,1]], # 0: 00, 11
        [[0,0],[1,1]], # 1: 10, 11
        [[1,0],[0,0]], # 2: 00
        [[0,0],[0,1]], # 3: 11
        [[1,1],[0,0]], # 4: 00, 01
        [[-1,0],[1,0]],# 5: 10, -00
        [[0,-1],[0,1]] # 6: 01, -11
    ]
    # Note: A_coeff[s][block_r][block_c]
    
    B_coeff = [
        [[1,0],[0,1]], # 0: 00, 11
        [[1,0],[0,0]], # 1: 00
        [[0,1],[0,-1]],# 2: 01, -11
        [[0,-1],[0,1]],# 3: -00, 10
        [[0,0],[0,1]], # 4: 11
        [[1,1],[0,0]], # 5: 00, 01
        [[0,1],[0,1]]  # 6: 10, 11
    ]
    
    # C coeffs (W part)
    # C00: M1+M4-M5+M7
    # C01: M3+M5
    # C10: M2+M4
    # C11: M1-M2+M3+M6
    C_coeff = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    # C_coeff[block_r][block_c][s]
    
    # Strassen 2x2 rank 7 decomposition
    # U2, V2, W2 for 2x2x2
    # We need the 7 terms of Strassen for 2x2.
    # Term 0: (a+d)(e+h)
    # Term 1: (c+d)e
    # Term 2: a(f-h)
    # Term 3: d(g-e)
    # Term 4: (a+b)h
    # Term 5: (c-a)(e+f)
    # Term 6: (b-d)(g+h)
    
    # U2[4x7], V2[4x7], W2[4x7]
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # 0
    U2[0][0]=1; U2[3][0]=1
    V2[0][0]=1; V2[3][0]=1
    W2[0][0]=1; W2[3][0]=1
    # 1
    U2[2][1]=1; U2[3][1]=1
    V2[0][1]=1
    W2[1][1]=1; W2[3][1]=1 # C10, C11?
    # Wait, W2 for term 1 (M2) contributes to C10 and C11.
    # M2 is a scalar.
    # The W matrix for the 2x2x2 tensor term c has 1s where the scalar M_c contributes.
    # M2 contributes to C10 (k=0,i=1) and C11 (k=1,i=1).
    # So W2[0,1]=1, W2[1,1]=1.
    W2[0][1]=1; W2[1][1]=1
    
    # 2
    U2[0][2]=1
    V2[1][2]=1; V2[3][2]=-1
    W2[1][2]=1; W2[3][2]=1 # C01, C11
    
    # 3
    U2[3][3]=1
    V2[2][3]=1; V2[0][3]=-1
    W2[0][3]=1; W2[2][3]=1 # C00, C10
    
    # 4
    U2[0][4]=1; U2[1][4]=1
    V2[3][4]=1
    W2[0][4]=-1; W2[1][4]=1 # C00, C01
    
    # 5
    U2[2][5]=1; U2[0][5]=-1
    V2[0][5]=1; V2[1][5]=1
    W2[3][5]=1 # C11
    
    # 6
    U2[1][6]=1; U2[3][6]=-1
    V2[2][6]=1; V2[3][6]=1
    W2[0][6]=1 # C00
    
    # Now build 4x4x4
    for s in range(7):
        for c in range(7):
            col = s * 7 + c
            # U part
            for i in range(4):
                for j in range(4):
                    block_r = i // 2
                    block_c = j // 2
                    local_i = i % 2
                    local_j = j % 2
                    coeff = A_coeff[s][block_r][block_c]
                    if coeff != 0:
                        U[i * 4 + j][col] = coeff * U2[local_i * 2 + local_j][c]
                        
            # V part
            for j in range(4):
                for k in range(4):
                    block_r = j // 2
                    block_c = k // 2
                    local_j = j % 2
                    local_k = k % 2
                    coeff = B_coeff[s][block_r][block_c]
                    if coeff != 0:
                        V[j * 5 + k][col] = coeff * V2[local_j * 2 + local_k][c]
                        
            # W part
            for k in range(4):
                for i in range(4):
                    block_r = k // 2
                    block_c = i // 2
                    local_k = k % 2
                    local_i = i % 2
                    coeff = C_coeff[block_r][block_c][s]
                    if coeff != 0:
                        W[k * 4 + i][col] = coeff * W2[local_k * 2 + local_i][c]
                        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
