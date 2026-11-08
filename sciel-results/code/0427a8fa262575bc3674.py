"""Matrix multiplication 2,3,3 with rank-17 decomposition."""

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 3
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 17
    V = [[0] * rank for _ in range(m * p)]  # 9 x 17
    W = [[0] * rank for _ in range(p * n)]  # 6 x 17
    
    # Sub-problem 1: A_1 (2x2) * B_11 (2x2) -> C_11 (2x2)
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U index: 2*i + j (global 0..3)
    # V index: 3*j + k (global 0..4)
    # W index: 3*k + i (global 0..5)
    # 4 terms (rank 4)
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[2*i + j][col] = 1
                V[3*j + k][col] = 1
                W[3*k + i][col] = 1
                col += 1
    # After: col = 4
    
    # Sub-problem 2: A_2 (2x1) * B_21 (1x2) -> C_21 (2x2)
    # Indices: i in {0,1}, j=2, k in {0,1}
    # U index: 2*i + 2 (global 4..5)
    # V index: 3*2 + k = 6 + k (global 6..7)
    # W index: 3*k + i (global 0..5)
    # 2 terms (rank 2)
    for i in range(2):
        for k in range(2):
            U[2*i + 2][col] = 1
            V[6 + k][col] = 1
            W[3*k + i][col] = 1
            col += 1
    # After: col = 6
    
    # Sub-problem 3: A_1 (2x2) * B_12 (2x1) -> C_12 (2x1)
    # Indices: i in {0,1}, j in {0,1}, k=2
    # U index: 2*i + j (global 0..3)
    # V index: 3*j + 2 (global 2, 5)
    # W index: 3*2 + i = 6 + i (global 6..7)
    # 2 terms (rank 2)
    for i in range(2):
        for j in range(2):
            U[2*i + j][col] = 1
            V[3*j + 2][col] = 1
            W[6 + i][col] = 1
            col += 1
    # After: col = 8
    
    # Sub-problem 4: A_2 (2x1) * B_22 (1x1) -> C_22 (2x1)
    # Indices: i in {0,1}, j=2, k=2
    # U index: 2*i + 2 (global 4..5)
    # V index: 3*2 + 2 = 8 (global 8)
    # W index: 3*2 + i = 6 + i (global 6..7)
    # 2 terms (rank 2)
    for i in range(2):
        U[2*i + 2][col] = 1
        V[8][col] = 1
        W[6 + i][col] = 1
        col += 1
    # After: col = 10
    
    # Wait, I only used 10 columns but claimed rank 17. 
    # Let me re-examine. The decomposition approach:
    # M(2,3,3) = M(2,2,2) + M(2,1,2) + M(2,2,1) + M(2,1,1)
    # Rank = 7 + 4 + 4 + 2 = 17
    # But my explicit construction above only used 4+2+2+2 = 10 terms.
    # 
    # The issue: M(2,2,2) has rank 7, not 4. I need to use a rank-7 
    # decomposition for the 2x2x2 part.
    #
    # Let me restart with proper rank-7 for M(2,2,2).
    
    # Clear and restart
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # M(2,2,2) rank-7 decomposition (Coppersmith-like)
    # For multiplying 2x2 by 2x2, we need 7 multiplications.
    # Standard Strassen-style:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # 
    # 7-term scheme:
    # m0 = (a+d)(e+h)
    # m1 = (c+d)e
    # m2 = a(f-h)
    # m3 = d(g-e)
    # m4 = (a+b)h
    # m5 = (c-d)(g+h)
    # m6 = (b-c)(e+g)
    #
    # C[0,0] = m0 + m3 - m4 + m6
    # C[0,1] = m2 + m4
    # C[1,0] = m1 + m5
    # C[1,1] = m0 - m1 + m2 + m5
    #
    # Tensor decomposition: each m_r contributes to U, V, W.
    # U[i,j] picks elements of A, V[j,k] picks elements of B, W[k,i] picks elements of C.
    #
    # For each of the 7 multiplications, we have a "coefficient" that may be +/-1
    # or 0 for each (i,j,k) combination.
    
    # Let me define the 7-term M(2,2,2) decomposition properly.
    # Using the standard 7-term scheme for 2x2 matrix multiplication:
    # 
    # Term 0: m0 = (a+d)(e+h)
    #   U: a->(0,0)=1, d->(1,1)=1 => U[0][0]=1, U[3][0]=1
    #   V: e->(0,0)=1, h->(1,1)=1 => V[0][0]=1, V[3][0]=1
    #   W: contribution to C[0,0]=+1, C[1,1]=+1 => W[0][0]=1, W[4][0]=1
    #
    # Term 1: m1 = (c+d)e
    #   U: c->(1,0)=1, d->(1,1)=1 => U[2][1]=1, U[3][1]=1
    #   V: e->(0,0)=1 => V[0][1]=1
    #   W: C[1,0]=+1, C[1,1]=-1 => W[2][1]=1, W[3][1]=-1
    #
    # Term 2: m2 = a(f-h)
    #   U: a->(0,0)=1 => U[0][2]=1
    #   V: f->(0,1)=1, -h->(1,1)=-1 => V[1][2]=1, V[3][2]=-1
    #   W: C[0,1]=+1, C[1,1]=+1 => W[1][2]=1, W[3][2]=1
    #
    # Term 3: m3 = d(g-e)
    #   U: d->(1,1)=1 => U[3][3]=1
    #   V: g->(1,0)=1, -e->(0,0)=-1 => V[2][3]=1, V[0][3]=-1
    #   W: C[0,0]=-1 => W[0][3]=-1
    #
    # Term 4: m4 = (a+b)h
    #   U: a->(0,0)=1, b->(0,1)=1 => U[0][4]=1, U[1][4]=1
    #   V: h->(1,1)=1 => V[3][4]=1
    #   W: C[0,0]=-1, C[0,1]=+1 => W[0][4]=-1, W[1][4]=1
    #
    # Term 5: m5 = (c-d)(g+h)
    #   U: c->(1,0)=1, -d->(1,1)=-1 => U[2][5]=1, U[3][5]=-1
    #   V: g->(1,0)=1, h->(1,1)=1 => V[2][5]=1, V[3][5]=1
    #   W: C[1,0]=+1, C[1,1]=+1 => W[2][5]=1, W[3][5]=1
    #
    # Term 6: m6 = (b-c)(e+g)
    #   U: b->(0,1)=1, -c->(1,0)=-1 => U[1][6]=1, U[2][6]=-1
    #   V: e->(0,0)=1, g->(1,0)=1 => V[0][6]=1, V[2][6]=1
    #   W: C[0,0]=+1 => W[0][6]=1
    
    # Global indices for sub-problem 1 (A_1: rows 0,1; B_11: cols 0,1):
    # U local (i,j) -> global 2*i+j, so i=0,j=0 -> 0; i=0,j=1 -> 1; i=1,j=0 -> 2; i=1,j=1 -> 3
    # V local (j,k) -> global 3*j+k, so j=0,k=0 -> 0; j=0,k=1 -> 1; j=1,k=0 -> 2; j=1,k=1 -> 3
    # W local (k,i) -> global 3*k+i, so k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
    
    # But wait, W has 6 rows (p*n = 3*2 = 6), indexed by 3*k+i where k in {0,1,2}, i in {0,1}.
    # For sub-problem 1, k in {0,1}, so W indices are 3*0+0=0, 3*0+1=1, 3*1+0=2, 3*1+1=3.
    
    # Term 0: m0
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1: m1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Term 2: m2
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Term 3: m3
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1
    
    # Term 4: m4
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 5: m5
    U[2][5] = 1; U[3][5] = -1
    V[2][5] = 1; V[3][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # Term 6: m6
    U[1][6] = 1; U[2][6] = -1
    V[0][6] = 1; V[2][6] = 1
    W[0][6] = 1
    
    # Sub-problem 2: A_2 (2x1) * B_21 (1x2) -> C_21 (2x2)
    # A_2: rows i in {0,1}, col j=2
    # B_21: row j=2, cols k in {0,1}
    # C_21: cols k in {0,1}, rows i in {0,1}
    #
    # U global: 2*i + 2 -> i=0 -> 4, i=1 -> 5
    # V global: 3*2 + k = 6+k -> k=0 -> 6, k=1 -> 7
    # W global: 3*k + i -> (k=0,i=0)->0, (k=0,i=1)->1, (k=1,i=0)->2, (k=1,i=1)->3
    #
    # This is M(2,1,2). Rank can be 4 (schoolbook: 2*1*2=4).
    # Terms: (i=0,k=0), (i=0,k=1), (i=1,k=0), (i=1,k=1)
    
    # Term 7: i=0, k=0
    U[4][7] = 1
    V[6][7] = 1
    W[0][7] = 1
    
    # Term 8: i=0, k=1
    U[4][8] = 1
    V[7][8] = 1
    W[1][8] = 1
    
    # Term 9: i=1, k=0
    U[5][9] = 1
    V[6][9] = 1
    W[2][9] = 1
    
    # Term 10: i=1, k=1
    U[5][10] = 1
    V[7][10] = 1
    W[3][10] = 1
    
    # Sub-problem 3: A_1 (2x2) * B_12 (2x1) -> C_12 (2x1)
    # A_1: rows i in {0,1}, cols j in {0,1}
    # B_12: rows j in {0,1}, col k=2
    # C_12: col k=2, rows i in {0,1}
    #
    # U global: 2*i + j -> i=0,j=0->0; i=0,j=1->1; i=1,j=0->2; i=1,j=1->3
    # V global: 3*j + 2 -> j=0->2, j=1->5
    # W global: 3*2 + i = 6+i -> i=0->6, i=1->7
    #
    # This is M(2,2,1). Rank can be 4 (schoolbook: 2*2*1=4).
    # Terms: (i=0,j=0), (i=0,j=1), (i=1,j=0), (i=1,j=1)
    
    # Term 11: i=0, j=0
    U[0][11] = 1
    V[2][11] = 1
    W[6][11] = 1
    
    # Term 12: i=0, j=1
    U[1][12] = 1
    V[5][12] = 1
    W[6][12] = 1
    
    # Term 13: i=1, j=0
    U[2][13] = 1
    V[2][13] = 1
    W[7][13] = 1
    
    # Term 14: i=1, j=1
    U[3][14] = 1
    V[5][14] = 1
    W[7][14] = 1
    
    # Sub-problem 4: A_2 (2x1) * B_22 (1x1) -> C_22 (2x1)
    # A_2: rows i in {0,1}, col j=2
    # B_22: row j=2, col k=2
    # C_22: col k=2, rows i in {0,1}
    #
    # U global: 2*i + 2 -> i=0->4, i=1->5
    # V global: 3*2+2 = 8
    # W global: 3*2 + i = 6+i -> i=0->6, i=1->7
    #
    # This is M(2,1,1). Rank = 2.
    
    # Term 15: i=0
    U[4][15] = 1
    V[8][15] = 1
    W[6][15] = 1
    
    # Term 16: i=1
    U[5][16] = 1
    V[8][16] = 1
    W[7][16] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
