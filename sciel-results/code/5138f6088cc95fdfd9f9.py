"""Program for matrix multiplication 2x2 * 2x14 using Strassen on 7 blocks."""

def construct():
    n, m, p = 2, 2, 14
    # p must be divisible by 2 for this block approach
    # Each block is 2x2, there are p//2 = 7 blocks
    
    # Strassen's 7 products for 2x2 multiplication
    # For matrices A, B (each 2x2):
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B10)
    # M7 = (A01 - A11)(B01 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U matrix: size (n*m, rank) = (4, rank)
    # V matrix: size (m*p, rank) = (28, rank)
    # W matrix: size (p*n, rank) = (28, rank)
    
    # rank = 7 * 7 = 49 (7 blocks, 7 Strassen steps each)
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # U_s[i,j] where s is the Strassen step index
    # A is 2x2, indexed by [i,j]
    U_steps = [
        # M1 = (A00 + A11)(B00 + B11)
        [[1, 0], [0, 1]],       # A00 + A11
        # M2 = (A10 + A11)B00
        [[0, 0], [1, 1]],       # A10 + A11
        # M3 = A00(B10 - B11)
        [[1, 0], [0, 0]],       # A00
        # M4 = A11(B01 - B00)
        [[0, 0], [0, 1]],       # A11
        # M5 = (A00 + A01)B11
        [[1, 1], [0, 0]],       # A00 + A01
        # M6 = (A10 - A00)(B00 + B10)
        [[-1, 0], [1, 0]],      # A10 - A00
        # M7 = (A01 - A11)(B01 + B11)
        [[0, 1], [0, -1]],      # A01 - A11
    ]
    
    # V_s[j,l] where s is Strassen step, j is row of B (0 or 1), l is col of B (0 or 1)
    V_steps = [
        # M1 = (A00 + A11)(B00 + B11)
        [[1, 1]],       # B00 + B11: row 0 has B00 at l=0, B01 at l=1... wait
        # Actually B is 2x2: B[j,l]
        # M1: (B00 + B11) means B[0,0] + B[1,1]
        # V_s[j,l]: coefficient of B[j,l]
        # For M1: B00 + B11 => V[0,0]=1, V[1,1]=1
        # Let me redo this properly.
        
        # V_s is a 2x2 matrix where V_s[j,l] is the coefficient of B[j,l]
        # M1 = (A00 + A11)(B00 + B11): V1[0,0]=1, V1[1,1]=1, others 0
        # M2 = (A10 + A11)B00: V2[0,0]=1, others 0
        # M3 = A00(B10 - B11): V3[1,0]=1, V3[1,1]=-1
        # M4 = A11(B01 - B00): V4[0,1]=1, V4[0,0]=-1
        # M5 = (A00 + A01)B11: V5[1,1]=1
        # M6 = (A10 - A00)(B00 + B10): V6[0,0]=1, V6[1,0]=1
        # M7 = (A01 - A11)(B01 + B11): V7[0,1]=1, V7[1,1]=1
    ]
    
    # Let me define V_steps properly as 2x2 lists
    V_steps = [
        [[1, 0],   # M1: B00 + B11
         [0, 1]],
        [[1, 0],   # M2: B00
         [0, 0]],
        [[0, 0],   # M3: B10 - B11
         [1, -1]],
        [[-1, 1],  # M4: B01 - B00
         [0, 0]],
        [[0, 0],   # M5: B11
         [0, 1]],
        [[1, 0],   # M6: B00 + B10
         [1, 0]],
        [[0, 1],   # M7: B01 + B11
         [0, 1]]
    ]
    
    # W_s[l,i] where l is the column index within the block (0 or 1), i is the output row (0 or 1)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # W_s[l,i]: coefficient of M_s in C[i, l_block] where l_block is 0 or 1 (within the 2-col block)
    
    # For each Strassen step s, W_s[l,i] gives contribution to C[i, block_col + l]
    W_steps = [
        # M1: C00 += 1, C11 += 1
        # l=0 (first col of block): C00 gets +1, C10 gets 0
        # l=1 (second col of block): C01 gets 0, C11 gets +1
        [[1, 0],   # l=0: i=0 -> +1 (C00), i=1 -> 0
         [0, 1]],  # l=1: i=0 -> 0, i=1 -> +1 (C11)
        # M2: C10 += 1
        [[0, 0],
         [0, 1]],  # l=0: i=1 -> +1 (C10)
        # M3: C01 += 1, C11 += 1
        [[0, 0],
         [1, 1]],  # l=1: i=0 -> +1 (C01), i=1 -> +1 (C11)
        # M4: C00 += 1, C10 += 1
        [[1, 0],
         [0, 1]],  # l=0: i=0 -> +1 (C00), i=1 -> +1 (C10)
        # M5: C00 -= 1, C01 += 1
        [[-1, 0],
         [1, 0]],  # l=0: i=0 -> -1 (C00), l=1: i=0 -> +1 (C01)
        # M6: C11 += 1
        [[0, 0],
         [0, 1]],  # l=1: i=1 -> +1 (C11)
        # M7: C00 += 1
        [[1, 0],
         [0, 0]],  # l=0: i=0 -> +1 (C00)
    ]
    
    # Now fill in U, V, W
    for b in range(p // 2):  # 7 blocks
        for s in range(7):  # 7 Strassen steps
            col = b * 7 + s
            
            # Fill U
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = U_steps[s][i][j]
            
            # Fill V
            for j in range(m):
                for l in range(2):
                    k_global = b * 2 + l
                    V[j * p + k_global][col] = V_steps[s][j][l]
            
            # Fill W
            for l in range(2):
                for i in range(n):
                    k_global = b * 2 + l
                    W[k_global * n + i][col] = W_steps[s][l][i]
    
    return {"U": U, "V": V, "W": W}
