import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition for rank 50.
    # Split B into B_12 (2x9) and B_3 (1x9).
    # Split A into A_12 (2x2) and A_3 (2x1).
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_12 * B_12 (2x2 * 2x9)
    # B_12 has 9 columns. Split into 4 blocks of 2 cols and 1 block of 1 col.
    # Blocks: k in [0,1], [2,3], [4,5], [6,7], [8]
    
    # Strassen coefficients for 2x2x2
    # M1..M7
    # U indices for A_12: i in {0,1}, j in {0,1} -> i*2+j
    # V indices for B_12: j in {0,1}, k in {0..8} -> j*9+k
    # W indices for C: k in {0..8}, i in {0,1} -> k*2+i
    
    # We will define the 7 multiplications for a 2x2x2 block.
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11 (local)
    # C indices: 00, 01, 10, 11 (local)
    
    strassen_U = [
        [1, 0, 1, 0], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [1, 0, -1, 0],# M6: A10 - A00
        [0, 1, 0, -1] # M7: A01 - A11
    ]
    
    strassen_V = [
        [1, 0, 1, 0], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01 - B11
        [0, -1, 1, 0],# M4: B10 - B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1]  # M7: B10 + B11
    ]
    
    strassen_W = [
        [1, 0, 0, -1], # C00: M1 - M2? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # Let's map M1..M7 to columns.
        # W is p*n x rank. Here p=2, n=2.
        # W[k*n+i, t]
    ]
    
    # Let's just hardcode the Strassen block for 2x2x2
    # U (4x7), V (4x7), W (4x7)
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    U_s = np.array([
        [1, 0, 1, 0, 1, 1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1], # A01
        [0, 1, 0, 0, 0, -1, 0],# A10
        [1, 1, 0, 1, 0, 0, -1] # A11
    ])
    
    V_s = np.array([
        [1, 1, 0, 0, 0, 1, 0], # B00
        [0, 0, 1, -1, 0, 1, 0],# B01
        [0, 0, 0, 1, 0, 0, 1], # B10
        [1, 0, -1, 0, 1, 0, 1] # B11
    ])
    
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ])
    
    # Verify Strassen
    # C00 = M1 + M4 - M5 + M7
    # M1 = (A00+A11)(B00+B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M7 = (A01-A11)(B10+B11)
    # Sum:
    # A00B00 + A00B11 + A11B00 + A11B11
    # + A11B10 - A11B00
    # - A00B11 - A01B11
    # + A01B10 + A01B11 - A11B10 - A11B11
    # = A00B00 + A11B10 - A01B11 + A01B10
    # Wait, C00 should be A00B00 + A01B10.
    # My W_s row 0: [1, 0, 0, 1, -1, 0, 1]
    # M1 - M5 + M7 + M4?
    # Let's re-verify the standard Strassen W matrix.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # My W_s:
    # Row 0 (C00): M1(1) + M4(1) - M5(-1) + M7(1) -> [1, 0, 0, 1, -1, 0, 1] Correct.
    # Row 1 (C01): M3(1) + M5(1) -> [0, 0, 1, 0, 1, 0, 0] Correct.
    # Row 2 (C10): M2(1) + M4(1) -> [0, 1, 0, 1, 0, 0, 0] Correct.
    # Row 3 (C11): M1(1) - M2(-1) + M3(1) + M6(1) -> [1, -1, 1, 0, 0, 1, 0] Correct.
    
    # Now apply to 4 blocks of 2x2x2 and 1 block of 2x2x1.
    
    # Block 1: k=0,1
    for t in range(7):
        for i in range(2):
            for j in range(2):
                U[i*3+j][col] = U_s[i,j][t] # A indices: i*3+j? No, A is 2x3.
                # A_12 uses j=0,1.
                # U index is i*m+j. m=3.
                # So i*3+0, i*3+1.
                U[i*3+j][col] = U_s[i,j][t]
        for j in range(2):
            for k in range(2):
                # B_12 uses j=0,1. k=0,1.
                # V index is j*p+k. p=9.
                # So j*9+k.
                V[j*9+k][col] = V_s[j,k][t]
        for k in range(2):
            for i in range(2):
                # C index is k*n+i. n=2.
                # So k*2+i.
                W[k*2+i][col] = W_s[k,i][t]
        col += 1
        
    # Block 2: k=2,3
    for t in range(7):
        for i in range(2):
            for j in range(2):
                U[i*3+j][col] = U_s[i,j][t]
        for j in range(2):
            for k in range(2):
                V[j*9+(2+k)][col] = V_s[j,k][t]
        for k in range(2):
            for i in range(2):
                W[(2+k)*2+i][col] = W_s[k,i][t]
        col += 1
        
    # Block 3: k=4,5
    for t in range(7):
        for i in range(2):
            for j in range(2):
                U[i*3+j][col] = U_s[i,j][t]
        for j in range(2):
            for k in range(2):
                V[j*9+(4+k)][col] = V_s[j,k][t]
        for k in range(2):
            for i in range(2):
                W[(4+k)*2+i][col] = W_s[k,i][t]
        col += 1
        
    # Block 4: k=6,7
    for t in range(7):
        for i in range(2):
            for j in range(2):
                U[i*3+j][col] = U_s[i,j][t]
        for j in range(2):
            for k in range(2):
                V[j*9+(6+k)][col] = V_s[j,k][t]
        for k in range(2):
            for i in range(2):
                W[(6+k)*2+i][col] = W_s[k,i][t]
        col += 1
        
    # Block 5: k=8 (2x2x1)
    # Rank 4.
    # C00 = A00B00 + A01B10
    # C01 = A10B00 + A11B10
    # Here B is 2x1. B00 is B[0,8], B10 is B[1,8].
    # A00 is A[0,0], A01 is A[0,1], etc.
    
    # Term 1: A00 * B00 -> C00
    U[0*3+0][col] = 1
    V[0*9+8][col] = 1
    W[8*2+0][col] = 1
    col += 1
    
    # Term 2: A01 * B10 -> C00
    U[0*3+1][col] = 1
    V[1*9+8][col] = 1
    W[8*2+0][col] = 1
    col += 1
    
    # Term 3: A10 * B00 -> C01
    U[1*3+0][col] = 1
    V[0*9+8][col] = 1
    W[8*2+1][col] = 1
    col += 1
    
    # Term 4: A11 * B10 -> C01
    U[1*3+1][col] = 1
    V[1*9+8][col] = 1
    W[8*2+1][col] = 1
    col += 1
    
    # Part 2: A_3 * B_3 (2x1 * 1x9)
    # A_3 is A[:, 2]. B_3 is B[2, :].
    # C_i,k = A_i,2 * B_2,k
    # 18 terms.
    for k in range(9):
        for i in range(2):
            U[i*3+2][col] = 1
            V[2*9+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
