import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 solution using Strassen blocks
    # A is 2x4, B is 4x6
    # Split A into A_L (2x2) and A_R (2x2)
    # Split B into B_T (2x6) and B_B (2x6)
    # C = A_L B_T + A_R B_B
    # B_T is 2x6, split into 3 blocks of 2x2
    # B_B is 2x6, split into 3 blocks of 2x2
    
    # Total 6 blocks of 2x2x2 multiplication
    # Each block rank 7
    # Total rank 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8x42
    V = [[0] * rank for _ in range(m * p)] # 24x42
    W = [[0] * rank for _ in range(p * n)] # 12x42
    
    # Strassen multipliers for 2x2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)h
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)g
    # M6 = (c-b)(e+h)
    # M7 = (b+d)(f-g)
    
    # Indices for A_L (rows 0,1 of A, cols 0,1 of A)
    # A is 2x4. Indices:
    # A[0,0]=a, A[0,1]=b, A[0,2]=e, A[0,3]=f (for A_R)
    # A[1,0]=c, A[1,1]=d, A[1,2]=g, A[1,3]=h (for A_R)
    
    # Indices for B_T (rows 0,1 of B, cols 0..5 of B)
    # B is 4x6. Indices:
    # B[0,0]=p0, B[0,1]=p1, B[0,2]=p2, B[0,3]=p3, B[0,4]=p4, B[0,5]=p5
    # B[1,0]=q0, B[1,1]=q1, B[1,2]=q2, B[1,3]=q3, B[1,4]=q4, B[1,5]=q5
    # B[2,0]=r0, ... (for B_B)
    # B[3,0]=s0, ...
    
    # Block 1: A_L * B_T_block1 (cols 0,1)
    # A_L: a,b,c,d
    # B_T1: p0,p1,q0,q1
    # Mults:
    # M1 = (a+d)(p0+q1) -> U[a]+U[d], V[p0]+V[q1]
    # M2 = (c+d)q1 -> U[c]+U[d], V[q1]
    # M3 = a(p1-q1) -> U[a], V[p1]-V[q1]
    # M4 = d(q0-p0) -> U[d], V[q0]-V[p0]
    # M5 = (a+b)q0 -> U[a]+U[b], V[q0]
    # M6 = (c-b)(p0+q1) -> U[c]-U[b], V[p0]+V[q1]
    # M7 = (b+d)(p1-q0) -> U[b]+U[d], V[p1]-V[q0]
    
    col = 0
    # Helper to add Strassen block
    def add_strassen_block(a_idx, b_idx, c_idx, d_idx, e_idx, f_idx, g_idx, h_idx, start_col):
        # a,b,c,d are A indices
        # e,f are B row 1 indices (top)
        # g,h are B row 2 indices (bottom)
        # Wait, B_T is 2x6. Rows 0,1 of B.
        # So e,f are B[0, col1], B[0, col2]
        # g,h are B[1, col1], B[1, col2]
        
        # M1 = (a+d)(e+h)
        U[a_idx][start_col + 0] = 1
        U[d_idx][start_col + 0] = 1
        V[e_idx][start_col + 0] = 1
        V[h_idx][start_col + 0] = 1
        
        # M2 = (c+d)h
        U[c_idx][start_col + 1] = 1
        U[d_idx][start_col + 1] = 1
        V[h_idx][start_col + 1] = 1
        
        # M3 = a(f-h)
        U[a_idx][start_col + 2] = 1
        V[f_idx][start_col + 2] = 1
        V[h_idx][start_col + 2] = -1
        
        # M4 = d(g-e)
        U[d_idx][start_col + 3] = 1
        V[g_idx][start_col + 3] = 1
        V[e_idx][start_col + 3] = -1
        
        # M5 = (a+b)g
        U[a_idx][start_col + 4] = 1
        U[b_idx][start_col + 4] = 1
        V[g_idx][start_col + 4] = 1
        
        # M6 = (c-b)(e+h)
        U[c_idx][start_col + 5] = 1
        U[b_idx][start_col + 5] = -1
        V[e_idx][start_col + 5] = 1
        V[h_idx][start_col + 5] = 1
        
        # M7 = (b+d)(f-g)
        U[b_idx][start_col + 6] = 1
        U[d_idx][start_col + 6] = 1
        V[f_idx][start_col + 6] = 1
        V[g_idx][start_col + 6] = -1
        
        # W contributions
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M6 + M7
        
        # W indices:
        # C is 2x6.
        # Block 1 affects C[0,0], C[0,1], C[1,0], C[1,1]
        # W[k*n+i, r]
        # C[0,0] is k=0, i=0 -> W[0]
        # C[0,1] is k=1, i=0 -> W[1]
        # C[1,0] is k=0, i=1 -> W[2]
        # C[1,1] is k=1, i=1 -> W[3]
        
        # M1
        W[0][start_col + 0] = 1
        W[3][start_col + 0] = 1
        # M2
        W[2][start_col + 1] = 1
        W[3][start_col + 1] = -1
        # M3
        W[1][start_col + 2] = 1
        # M4
        W[0][start_col + 3] = 1
        W[2][start_col + 3] = 1
        # M5
        W[0][start_col + 4] = -1
        W[1][start_col + 4] = 1
        # M6
        W[3][start_col + 5] = 1
        # M7
        W[0][start_col + 6] = 1
        W[3][start_col + 6] = 1
        
        return start_col + 7

    # A_L indices:
    # a=0, b=1, c=4, d=5 (A is 2x4, row-major: 0,1,2,3 / 4,5,6,7)
    # A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    
    # B_T indices (rows 0,1 of B):
    # B is 4x6. Row 0: 0..5. Row 1: 6..11.
    # Block 1: cols 0,1 -> B[0,0]=0, B[0,1]=1, B[1,0]=6, B[1,1]=7
    # Block 2: cols 2,3 -> B[0,2]=2, B[0,3]=3, B[1,2]=8, B[1,3]=9
    # Block 3: cols 4,5 -> B[0,4]=4, B[0,5]=5, B[1,4]=10, B[1,5]=11
    
    # A_R indices:
    # e=2, f=3, g=6, h=7
    
    # B_B indices (rows 2,3 of B):
    # Row 2: 12..17. Row 3: 18..23.
    # Block 1: cols 0,1 -> B[2,0]=12, B[2,1]=13, B[3,0]=18, B[3,1]=19
    # Block 2: cols 2,3 -> B[2,2]=14, B[2,3]=15, B[3,2]=20, B[3,3]=21
    # Block 3: cols 4,5 -> B[2,4]=16, B[2,5]=17, B[3,4]=22, B[3,5]=23
    
    # W indices for C (2x6):
    # C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3, C[0,4]=4, C[0,5]=5
    # C[1,0]=6, C[1,1]=7, C[1,2]=8, C[1,3]=9, C[1,4]=10, C[1,5]=11
    
    # The add_strassen_block function above assumed W indices 0..3 for the 2x2 block.
    # I need to offset W indices for each block.
    
    def add_strassen_block_offset(a_idx, b_idx, c_idx, d_idx, e_idx, f_idx, g_idx, h_idx, w_offset, start_col):
        # M1
        U[a_idx][start_col + 0] = 1
        U[d_idx][start_col + 0] = 1
        V[e_idx][start_col + 0] = 1
        V[h_idx][start_col + 0] = 1
        W[w_offset + 0][start_col + 0] = 1
        W[w_offset + 3][start_col + 0] = 1
        
        # M2
        U[c_idx][start_col + 1] = 1
        U[d_idx][start_col + 1] = 1
        V[h_idx][start_col + 1] = 1
        W[w_offset + 2][start_col + 1] = 1
        W[w_offset + 3][start_col + 1] = -1
        
        # M3
        U[a_idx][start_col + 2] = 1
        V[f_idx][start_col + 2] = 1
        V[h_idx][start_col + 2] = -1
        W[w_offset + 1][start_col + 2] = 1
        
        # M4
        U[d_idx][start_col + 3] = 1
        V[g_idx][start_col + 3] = 1
        V[e_idx][start_col + 3] = -1
        W[w_offset + 0][start_col + 3] = 1
        W[w_offset + 2][start_col + 3] = 1
        
        # M5
        U[a_idx][start_col + 4] = 1
        U[b_idx][start_col + 4] = 1
        V[g_idx][start_col + 4] = 1
        W[w_offset + 0][start_col + 4] = -1
        W[w_offset + 1][start_col + 4] = 1
        
        # M6
        U[c_idx][start_col + 5] = 1
        U[b_idx][start_col + 5] = -1
        V[e_idx][start_col + 5] = 1
        V[h_idx][start_col + 5] = 1
        W[w_offset + 3][start_col + 5] = 1
        
        # M7
        U[b_idx][start_col + 6] = 1
        U[d_idx][start_col + 6] = 1
        V[f_idx][start_col + 6] = 1
        V[g_idx][start_col + 6] = -1
        W[w_offset + 0][start_col + 6] = 1
        W[w_offset + 3][start_col + 6] = 1
        
        return start_col + 7

    col = 0
    # A_L blocks
    # Block 1
    col = add_strassen_block_offset(0, 1, 4, 5, 0, 1, 6, 7, 0, col)
    # Block 2
    col = add_strassen_block_offset(0, 1, 4, 5, 2, 3, 8, 9, 2, col)
    # Block 3
    col = add_strassen_block_offset(0, 1, 4, 5, 4, 5, 10, 11, 4, col)
    
    # A_R blocks
    # Block 1
    col = add_strassen_block_offset(2, 3, 6, 7, 12, 13, 18, 19, 0, col)
    # Block 2
    col = add_strassen_block_offset(2, 3, 6, 7, 14, 15, 20, 21, 2, col)
    # Block 3
    col = add_strassen_block_offset(2, 3, 6, 7, 16, 17, 22, 23, 4, col)
    
    return {"U": U, "V": V, "W": W}
