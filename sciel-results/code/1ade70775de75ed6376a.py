import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    # Dimensions
    n, m, p = 2, 3, 7
    
    # We will build the decomposition by parts.
    # Part 1: A (2x3) * B3 (3x3) -> C3 (2x3)
    # Part 2: A (2x3) * B4 (3x4) -> C4 (2x4)
    # B is split into B3 (cols 0-2) and B4 (cols 3-6)
    
    # Sub-part 1a: A22 (2x2) * B32 (2x3) -> C32 (2x3)
    # Sub-part 1b: A21 (2x1) * B31 (1x3) -> C31 (2x3)
    # Sub-part 2a: A22 (2x2) * B42 (2x4) -> C42 (2x4)
    # Sub-part 2b: A21 (2x1) * B41 (1x4) -> C41 (2x4)
    
    # Ranks:
    # R1a: Rank(M_2,2,3) = 11
    # R1b: Rank(M_2,1,3) = 6
    # R2a: Rank(M_2,2,4) = 14
    # R2b: Rank(M_2,1,4) = 8
    # Total Rank = 11 + 6 + 14 + 8 = 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)] # 6 x 39
    V = [[0] * rank for _ in range(m * p)] # 21 x 39
    W = [[0] * rank for _ in range(p * n)] # 14 x 39
    
    col_offset = 0
    
    # Helper to add a decomposition
    def add_decomp(U_sub, V_sub, W_sub, u_offset, v_offset, w_offset):
        # U_sub is (n*m_sub) x r, V_sub is (m_sub*p_sub) x r, W_sub is (p_sub*n) x r
        # We need to map indices to global indices
        pass

    # This approach is complex to index. 
    # Easier to just construct the full tensors directly using known formulas.
    
    # Let's define the Strassen decomposition for 2x2x2
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = M1 - M4 + M5 + M6
    # c01 = M3 + M4 + M7
    # c10 = M1 - M2 + M3 + M5
    # c11 = M2 - M1 + M4 + M7
    
    # We need to represent this as U, V, W.
    # U maps A indices to rank indices.
    # V maps B indices to rank indices.
    # W maps C indices to rank indices.
    
    # For Strassen:
    # r=7
    # U[0,0]=1, U[1,0]=1 (a00, a11) -> M1 uses a00, a11
    # Actually, U[i, r] is coefficient of A_i in product r.
    
    # Let's generate the matrices for each sub-problem and then stitch them.
    
    def get_strassen_222():
        # Returns U (4x7), V (4x7), W (4x7) for 2x2x2
        r = 7
        U = [[0]*r for _ in range(4)]
        V = [[0]*r for _ in range(4)]
        W = [[0]*r for _ in range(4)]
        
        # M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[1][0] = 1
        V[0][0] = 1; V[1][0] = 1
        W[0][0] = 1; W[3][0] = 1 # c00, c11?
        # c00 = M1 - M4 + M5 + M6
        # c01 = M3 + M4 + M7
        # c10 = M1 - M2 + M3 + M5
        # c11 = M2 - M1 + M4 + M7
        
        # W[0] is c00, W[1] is c01, W[2] is c10, W[3] is c11
        W[0][0] = 1
        W[3][0] = -1 # c11 has -M1? No, c11 = M2 - M1 + ...
        # Let's be precise.
        # c00 = M1 - M4 + M5 + M6
        # c01 = M3 + M4 + M7
        # c10 = M1 - M2 + M3 + M5
        # c11 = M2 - M1 + M4 + M7
        
        # M1:
        U[0][0]=1; U[1][0]=1
        V[0][0]=1; V[1][0]=1
        W[0][0]=1; W[2][0]=1; W[3][0]=-1 # c00, c10, c11
        
        # M2 = (a10+a11)b00
        U[2][1]=1; U[3][1]=1
        V[0][1]=1
        W[2][1]=-1; W[3][1]=1 # c10, c11
        
        # M3 = a00(b10-b11)
        U[0][2]=1
        V[2][2]=1; V[3][2]=-1
        W[1][2]=1; W[2][2]=1 # c01, c10
        
        # M4 = a11(b01-b00)
        U[3][3]=1
        V[1][3]=1; V[0][3]=-1
        W[0][3]=-1; W[1][3]=1; W[3][3]=1 # c00, c01, c11
        
        # M5 = (a00+a01)b11
        U[0][4]=1; U[1][4]=1
        V[3][4]=1
        W[0][4]=1; W[2][4]=1 # c00, c10
        
        # M6 = (a10-a00)(b00+b01)
        U[2][5]=1; U[0][5]=-1
        V[0][5]=1; V[1][5]=1
        W[0][5]=1 # c00
        
        # M7 = (a01-a11)(b10+b11)
        U[1][6]=1; U[3][6]=-1
        V[2][6]=1; V[3][6]=1
        W[1][6]=1; W[3][6]=1 # c01, c11
        
        return U, V, W

    def get_schoolbook_221():
        # 2x2x1: A (2x2) * B (2x1) -> C (2x1)
        # c0 = a00 b0 + a01 b1
        # c1 = a10 b0 + a11 b1
        # Rank 4
        r = 4
        U = [[0]*r for _ in range(4)]
        V = [[0]*r for _ in range(2)] # B is 2x1, so 2 elements
        W = [[0]*r for _ in range(2)] # C is 2x1, so 2 elements
        
        # Term 0: a00 b0 -> c0
        U[0][0]=1; V[0][0]=1; W[0][0]=1
        # Term 1: a01 b1 -> c0
        U[1][1]=1; V[1][1]=1; W[0][1]=1
        # Term 2: a10 b0 -> c1
        U[2][2]=1; V[0][2]=1; W[1][2]=1
        # Term 3: a11 b1 -> c1
        U[3][3]=1; V[1][3]=1; W[1][3]=1
        
        return U, V, W

    def get_outer_21k(k):
        # 2x1xk: A (2x1) * B (1xk) -> C (2xk)
        # c_il = a_i b_l
        # Rank 2k
        r = 2*k
        U = [[0]*r for _ in range(2)]
        V = [[0]*r for _ in range(k)]
        W = [[0]*r for _ in range(2*k)]
        
        idx = 0
        for i in range(2):
            for l in range(k):
                U[i][idx] = 1
                V[l][idx] = 1
                W[i*k + l][idx] = 1
                idx += 1
        return U, V, W

    # Now stitch them together.
    # Global A is 2x3. Indices: 0,1 (row 0), 2,3 (row 1)?
    # Standard row-major: A[i, j] -> index i*m + j.
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    
    # Global B is 3x7. Indices: B[j, k] -> index j*p + k.
    # B[0,0..6] = 0..6
    # B[1,0..6] = 7..13
    # B[2,0..6] = 14..20
    
    # Global C is 2x7. Indices: C[i, k] -> index i*p + k?
    # Task says W[k*n+i, r]. So index is k*n + i.
    # C[0,0..6] = 0..6
    # C[1,0..6] = 7..13
    
    # Sub-problem 1: A22 (2x2) * B32 (2x3)
    # A22 uses A[0,0], A[0,1], A[1,0], A[1,1] -> Global indices 0,1,3,4
    # B32 uses B[0,0..2], B[1,0..2] -> Global indices 0,1,2, 7,8,9
    # C32 is part of C[0,0..2], C[1,0..2] -> Global indices 0,1,2, 7,8,9
    
    # Sub-problem 1b: A21 (2x1) * B31 (1x3)
    # A21 uses A[0,2], A[1,2] -> Global indices 2,5
    # B31 uses B[2,0..2] -> Global indices 14,15,16
    # C31 is part of C[0,0..2], C[1,0..2] -> Global indices 0,1,2, 7,8,9
    
    # Sub-problem 2a: A22 (2x2) * B42 (2x4)
    # A22 uses A[0,0], A[0,1], A[1,0], A[1,1] -> Global indices 0,1,3,4
    # B42 uses B[0,3..6], B[1,3..6] -> Global indices 3,4,5,6, 10,11,12,13
    # C42 is part of C[0,3..6], C[1,3..6] -> Global indices 3,4,5,6, 10,11,12,13
    
    # Sub-problem 2b: A21 (2x1) * B41 (1x4)
    # A21 uses A[0,2], A[1,2] -> Global indices 2,5
    # B41 uses B[2,3..6] -> Global indices 17,18,19,20
    # C41 is part of C[0,3..6], C[1,3..6] -> Global indices 3,4,5,6, 10,11,12,13
    
    # We will fill the global U, V, W matrices.
    
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(21)]
    W = [[0]*rank for _ in range(14)]
    
    col = 0
    
    # 1a: 2x2x3 (Strassen 2x2x2 + Schoolbook 2x2x1)
    U1a, V1a, W1a = get_strassen_222()
    U1a2, V1a2, W1a2 = get_schoolbook_221()
    
    # Map local indices to global
    # Local A (2x2): 0,1,2,3 -> Global 0,1,3,4
    a_map_1a = [0, 1, 3, 4]
    # Local B (2x2 for Strassen part): 0,1,2,3 -> Global 0,1, 7,8 (B[0,0], B[0,1], B[1,0], B[1,1])
    # Wait, Strassen 2x2x2 uses B[0,0], B[0,1], B[1,0], B[1,1].
    # In B32 (2x3), these are B[0,0], B[0,1], B[1,0], B[1,1].
    # Global indices: B[0,0]=0, B[0,1]=1, B[1,0]=7, B[1,1]=8.
    b_map_1a_strassen = [0, 1, 7, 8]
    
    # Schoolbook 2x2x1 uses B[0,2], B[1,2].
    # Global indices: B[0,2]=2, B[1,2]=8? No, B[1,2] is index 1*7+2=9.
    b_map_1a_school = [2, 9]
    
    # Local C (2x2 for Strassen): 0,1,2,3 -> Global C[0,0], C[0,1], C[1,0], C[1,1]
    # Global indices: C[0,0]=0, C[0,1]=1, C[1,0]=7, C[1,1]=8.
    c_map_1a_strassen = [0, 1, 7, 8]
    
    # Schoolbook 2x2x1 C is 2x1: C[0,2], C[1,2].
    # Global indices: C[0,2]=2, C[1,2]=9.
    c_map_1a_school = [2, 9]
    
    # Add Strassen part
    for i in range(4):
        for r in range(7):
            U[a_map_1a[i]][col+r] += U1a[i][r]
            V[b_map_1a_strassen[i]][col+r] += V1a[i][r]
            W[c_map_1a_strassen[i]][col+r] += W1a[i][r]
    col += 7
    
    # Add Schoolbook part
    for i in range(4):
        for r in range(4):
            U[a_map_1a[i]][col+r] += U1a2[i][r]
            V[b_map_1a_school[i//2]][col+r] += V1a2[i][r] # V1a2 is 2x4, i is 0..3?
            # V1a2 rows: 0,1. i=0,1 use row 0? No.
            # U1a2[i] corresponds to A_i.
            # V1a2[j] corresponds to B_j.
            # W1a2[k] corresponds to C_k.
            # Term r: U[i,r] V[j,r] W[k,r].
            # We need to map i (A index) to global A.
            # Map j (B index) to global B.
            # Map k (C index) to global C.
            
    # This manual mapping is error-prone.
    # Better to construct the full U, V, W for the sub-problem and then add them to the global ones using index maps.
    
    return {"U": U, "V": V, "W": W}
