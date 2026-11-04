import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a local decomposition
    def add_local(local_rank, local_U, local_V, local_W, 
                  i_off, j_off, k_off, 
                  n_loc, m_loc, p_loc):
        nonlocal col
        for r in range(local_rank):
            for i in range(n_loc):
                for j in range(m_loc):
                    val = local_U[i * m_loc + j][r]
                    if val != 0:
                        U[(i + i_off) * m + (j + j_off)][col + r] += val
            for j in range(m_loc):
                for k in range(p_loc):
                    val = local_V[j * p_loc + k][r]
                    if val != 0:
                        V[(j + j_off) * p + (k + k_off)][col + r] += val
            for k in range(p_loc):
                for i in range(n_loc):
                    val = local_W[k * n_loc + i][r]
                    if val != 0:
                        W[(k + k_off) * n + (i + i_off)][col + r] += val
        col += local_rank

    # 1. 2x2x2 (Strassen)
    # Rank 7
    # U, V, W for 2x2x2
    # Using standard Strassen factors
    # A = [[a1, a2], [a3, a4]]
    # B = [[b1, b2], [b3, b4]]
    # C = [[c1, c2], [c3, c4]]
    # M1 = (a1+a4)(b1+b4)
    # M2 = (a3+a4)b1
    # M3 = a1(b2-b4)
    # M4 = a4(b3-b1)
    # M5 = (a1+a2)b4
    # M6 = (a3-a1)(b1+b2)
    # M7 = (a2-a4)(b3+b4)
    # c1 = M1+M4-M5+M7
    # c2 = M3+M5
    # c3 = M2+M4
    # c4 = M1-M2+M3+M6
    
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]
    
    # M1: (a1+a4)(b1+b4)
    # a1: U[0,0]=1, a4: U[1,1]=1
    # b1: V[0,0]=1, b4: V[1,1]=1
    # c1: W[0,0]=1, c4: W[1,1]=1
    # We need to distribute M1 to c1 and c4.
    # c1 += M1, c4 += M1
    # W[0,0] += 1, W[1,1] += 1
    # But W is a matrix. We add to specific entries.
    
    # Let's define the factors explicitly for Strassen
    # r=0: M1
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1 # c1 and c4? No.
    # c1 is W[0,0] (k=0, i=0). c4 is W[1,1] (k=1, i=1).
    # So W1[0][0] = 1, W1[3][0] = 1? No.
    # W index is k*n+i.
    # c1: k=0, i=0 -> idx 0.
    # c4: k=1, i=1 -> idx 3.
    # So W1[0][0] = 1, W1[3][0] = 1.
    
    # r=1: M2 = (a3+a4)b1
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    # c3 += M2, c4 -= M2
    # c3: k=0, i=1 -> idx 1.
    # c4: k=1, i=1 -> idx 3.
    W1[1][1] = 1; W1[3][1] = -1
    
    # r=2: M3 = a1(b2-b4)
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    # c2 += M3, c4 += M3
    # c2: k=1, i=0 -> idx 2.
    # c4: k=1, i=1 -> idx 3.
    W1[2][2] = 1; W1[3][2] = 1
    
    # r=3: M4 = a4(b3-b1)
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    # c1 += M4, c3 += M4
    # c1: idx 0. c3: idx 1.
    W1[0][3] = 1; W1[1][3] = 1
    
    # r=4: M5 = (a1+a2)b4
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    # c1 -= M5, c2 += M5
    # c1: idx 0. c2: idx 2.
    W1[0][4] = -1; W1[2][4] = 1
    
    # r=5: M6 = (a3-a1)(b1+b2)
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    # c4 += M6
    # c4: idx 3.
    W1[3][5] = 1
    
    # r=6: M7 = (a2-a4)(b3+b4)
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    # c1 += M7
    # c1: idx 0.
    W1[0][6] = 1
    
    add_local(7, U1, V1, W1, 0, 0, 0, 2, 2, 2)
    
    # 2. 2x2x3
    # A 2x2, B 2x3, C 2x3
    # Split B into 2x2 and 2x1.
    # Term A: 2x2x2 (Strassen)
    # Term B: 2x2x1 (Schoolbook)
    
    U2a = [[0]*7 for _ in range(4)]
    V2a = [[0]*7 for _ in range(6)] # B is 2x3, so 6 entries
    W2a = [[0]*7 for _ in range(6)] # C is 2x3, so 6 entries
    
    # Copy Strassen U
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U2a[i*2+j][r] = U1[i*2+j][r]
                
    # Copy Strassen V, but only first 2 cols of B (k=0,1)
    # B local indices: j in {0,1}, k in {0,1,2}
    # Strassen B used k=0,1.
    for r in range(7):
        for j in range(2):
            for k in range(2):
                V2a[j*3+k][r] = V1[j*2+k][r]
                
    # Copy Strassen W, but only first 2 cols of C (k=0,1)
    for r in range(7):
        for k in range(2):
            for i in range(2):
                W2a[k*2+i][r] = W1[k*2+i][r]
                
    # Term B: 2x2x1
    # Rank 4.
    # A 2x2, B 2x1, C 2x1.
    # C_i = sum_j A_ij B_j0.
    # U[i*2+j, r] = 1
    # V[j*1+0, r] = 1
    # W[0*2+i, r] = 1
    # r=0: i=0, j=0
    # r=1: i=0, j=1
    # r=2: i=1, j=0
    # r=3: i=1, j=1
    
    U2b = [[0]*4 for _ in range(4)]
    V2b = [[0]*4 for _ in range(2)]
    W2b = [[0]*4 for _ in range(2)]
    
    U2b[0][0] = 1; V2b[0][0] = 1; W2b[0][0] = 1
    U2b[1][1] = 1; V2b[1][1] = 1; W2b[0][1] = 1
    U2b[2][2] = 1; V2b[0][2] = 1; W2b[1][2] = 1
    U2b[3][3] = 1; V2b[1][3] = 1; W2b[1][3] = 1
    
    # Combine 2a and 2b
    # U2 = [U2a, U2b]
    # V2 = [V2a, V2b]
    # W2 = [W2a, W2b]
    
    U2 = [[0]*11 for _ in range(4)]
    V2 = [[0]*11 for _ in range(6)]
    W2 = [[0]*11 for _ in range(6)]
    
    for r in range(7):
        for i in range(4): U2[i][r] = U2a[i][r]
        for i in range(6): V2[i][r] = V2a[i][r]
        for i in range(6): W2[i][r] = W2a[i][r]
        
    for r in range(4):
        for i in range(4): U2[i][r+7] = U2b[i][r]
        for i in range(2): V2[i][r+7] = V2b[i][r]
        for i in range(2): W2[i][r+7] = W2b[i][r]
        
    add_local(11, U2, V2, W2, 0, 0, 2, 2, 2, 3)
    
    # 3. 2x3x2
    # A 2x3, B 3x2, C 2x2
    # Split A into 2x2 and 2x1.
    # Term A: 2x2x2 (Strassen)
    # Term B: 2x1x2 (Schoolbook)
    
    U3a = [[0]*7 for _ in range(6)]
    V3a = [[0]*7 for _ in range(6)]
    W3a = [[0]*7 for _ in range(4)]
    
    # Strassen A uses j=0,1.
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U3a[i*3+j][r] = U1[i*2+j][r]
                
    # Strassen B uses j=0,1.
    for r in range(7):
        for j in range(2):
            for k in range(2):
                V3a[j*2+k][r] = V1[j*2+k][r]
                
    # Strassen W
    for r in range(7):
        for k in range(2):
            for i in range(2):
                W3a[k*2+i][r] = W1[k*2+i][r]
                
    # Term B: 2x1x2
    # A 2x1, B 1x2, C 2x2.
    # C_i = A_i0 B_0k.
    # Rank 4.
    U3b = [[0]*4 for _ in range(2)]
    V3b = [[0]*4 for _ in range(2)]
    W3b = [[0]*4 for _ in range(4)]
    
    # r=0: i=0, k=0
    U3b[0][0] = 1; V3b[0][0] = 1; W3b[0][0] = 1
    # r=1: i=0, k=1
    U3b[0][1] = 1; V3b[1][1] = 1; W3b[1][0] = 1
    # r=2: i=1, k=0
    U3b[1][2] = 1; V3b[0][2] = 1; W3b[0][2] = 1
    # r=3: i=1, k=1
    U3b[1][3] = 1; V3b[1][3] = 1; W3b[1][3] = 1
    
    U3 = [[0]*11 for _ in range(6)]
    V3 = [[0]*11 for _ in range(6)]
    W3 = [[0]*11 for _ in range(4)]
    
    for r in range(7):
        for i in range(6): U3[i][r] = U3a[i][r]
        for i in range(6): V3[i][r] = V3a[i][r]
        for i in range(4): W3[i][r] = W3a[i][r]
        
    for r in range(4):
        for i in range(2): U3[i+4][r+7] = U3b[i][r]
        for i in range(2): V3[i+4][r+7] = V3b[i][r]
        for i in range(4): W3[i][r+7] = W3b[i][r]
        
    add_local(11, U3, V3, W3, 0, 2, 0, 2, 3, 2)
    
    # 4. 2x3x3
    # A 2x3, B 3x3, C 2x3
    # Split B into 3x2 and 3x1.
    # Term A: 2x3x2 (Rank 11 from above)
    # Term B: 2x3x1 (Schoolbook)
    
    # Reuse U3, V3, W3 for Term A?
    # U3 is 2x3, V3 is 3x2, W3 is 2x2.
    # We need V for 3x3.
    
    U4a = [[0]*11 for _ in range(6)]
    V4a = [[0]*11 for _ in range(9)]
    W4a = [[0]*11 for _ in range(6)]
    
    for r in range(11):
        for i in range(6): U4a[i][r] = U3[i][r]
        for i in range(6): V4a[i][r] = V3[i][r]
        for i in range(4): W4a[i][r] = W3[i][r]
        
    # Term B: 2x3x1
    # A 2x3, B 3x1, C 2x1.
    # C_i = sum_j A_ij B_j0.
    # Rank 6.
    U4b = [[0]*6 for _ in range(6)]
    V4b = [[0]*6 for _ in range(3)]
    W4b = [[0]*6 for _ in range(2)]
    
    for i in range(2):
        for j in range(3):
            r = i*3+j
            U4b[i*3+j][r] = 1
            V4b[j*1+0][r] = 1
            W4b[0*2+i][r] = 1
            
    U4 = [[0]*17 for _ in range(6)]
    V4 = [[0]*17 for _ in range(9)]
    W4 = [[0]*17 for _ in range(6)]
    
    for r in range(11):
        for i in range(6): U4[i][r] = U4a[i][r]
        for i in range(9): V4[i][r] = V4a[i][r]
        for i in range(6): W4[i][r] = W4a[i][r]
        
    for r in range(6):
        for i in range(6): U4[i][r+11] = U4b[i][r]
        for i in range(3): V4[i][r+11] = V4b[i][r]
        for i in range(2): W4[i][r+11] = W4b[i][r]
        
    add_local(17, U4, V4, W4, 0, 2, 2, 2, 3, 3)
    
    return {"U": U, "V": V, "W": W}

