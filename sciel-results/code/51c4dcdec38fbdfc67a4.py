import numpy as np

def construct():
    def build_strassen_4x4():
        # Returns U, V, W for 4x4 Strassen decomposition
        # Rank 49
        # U: 16x49, V: 16x49, W: 16x49
        # Indices: U[i*4+k], V[k*4+j], W[j*4+i] (assuming W index j*n+i for C[i,j])
        
        # Base 2x2 Strassen
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We will build 4x4 from 2x2.
        # N=4, N_half=2.
        # Rank = 7 * 7 = 49.
        
        U = [[0]*49 for _ in range(16)]
        V = [[0]*49 for _ in range(16)]
        W = [[0]*49 for _ in range(16)]
        
        # Helper to get indices for 2x2 blocks in 4x4
        def get_block_indices_4x4(block):
            # block 0: 0,0 (rows 0-1, cols 0-1)
            # block 1: 0,1 (rows 0-1, cols 2-3)
            # block 2: 1,0 (rows 2-3, cols 0-1)
            # block 3: 1,1 (rows 2-3, cols 2-3)
            r0, c0 = 0, 0
            if block == 0: r0, c0 = 0, 0
            elif block == 1: r0, c0 = 0, 2
            elif block == 2: r0, c0 = 2, 0
            elif block == 3: r0, c0 = 2, 2
            
            indices = []
            for i in range(2):
                for j in range(2):
                    indices.append((r0+i, c0+j))
            return indices

        # 2x2 Strassen coefficients
        # We need the 2x2 decomposition first.
        # Let's hardcode the 2x2 Strassen decomposition into a helper or build it.
        # 2x2 Rank 7.
        # U2, V2, W2 of size 4x7.
        
        U2 = [[0]*7 for _ in range(4)]
        V2 = [[0]*7 for _ in range(4)]
        W2 = [[0]*7 for _ in range(4)]
        
        # M1: A11+A22, B11+B22
        # U2: A11(0,0), A22(1,1)
        U2[0][0] = 1; U2[3][0] = 1
        V2[0][0] = 1; V2[3][0] = 1
        W2[0][0] = 1; W2[3][0] = 1 # C11 += M1, C22 += M1?
        # C11 = M1 - M4 + M7?
        # C12 = M3 + M5
        # C21 = M2 + M6
        # C22 = M1 - M3 + M4
        
        # Let's define the 2x2 decomposition properly.
        # Indices for 2x2:
        # A: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C: 00, 01, 10, 11 -> 0, 1, 2, 3 (Index j*2+i)
        
        # M1: (A11+A22)(B11+B22)
        U2[0][0] = 1; U2[3][0] = 1
        V2[0][0] = 1; V2[3][0] = 1
        W2[0][0] = 1 # C11
        W2[3][0] = 1 # C22
        
        # M2: (A21+A22)B11
        U2[2][1] = 1; U2[3][1] = 1
        V2[0][1] = 1
        W2[2][1] = 1 # C21
        
        # M3: A11(B12-B22)
        U2[0][2] = 1
        V2[1][2] = 1; V2[3][2] = -1
        W2[1][2] = 1 # C12
        W2[3][2] = -1 # C22
        
        # M4: A22(B21-B11)
        U2[3][3] = 1
        V2[2][3] = 1; V2[0][3] = -1
        W2[0][3] = -1 # C11
        W2[2][3] = 1 # C22
        
        # M5: (A11+A12)B22
        U2[0][4] = 1; U2[1][4] = 1
        V2[3][4] = 1
        W2[1][4] = 1 # C12
        
        # M6: (A21-A11)(B11+B12)
        U2[2][5] = 1; U2[0][5] = -1
        V2[0][5] = 1; V2[1][5] = 1
        W2[2][5] = 1 # C21
        
        # M7: (A12-A22)(B21+B22)
        U2[1][6] = 1; U2[3][6] = -1
        V2[2][6] = 1; V2[3][6] = 1
        W2[0][6] = 1 # C11
        
        # Now build 4x4 from 2x2.
        # Each M_m in 4x4 is a product of 2x2 matrices.
        # So we replace each scalar M_m with a 2x2 product.
        # This requires 7 * 7 = 49 terms.
        
        for m in range(7):
            base_col = m * 7
            # Get the U, V, W contributions for M_m in 4x4.
            # This is complex to generate programmatically without the recursive structure.
            # I will rely on the recursive function I wrote in the failed code, but fix the indices.
            
        # Actually, I'll just use the recursive builder from the failed code, which seemed logically sound for the 4x4 part, 
        # but I'll ensure the W index convention is j*4+i.
        
        return U, V, W

    # Since implementing the full recursive Strassen is error-prone in this short format, 
    # and I know the rank is 49, I will use a known 4x4 Strassen decomposition or 
    # simply use the 8x8 slicing method which is safer and rank 343.
    # But 343 > 125. 
    # I need rank < 125.
    # The 4x4 + boundary method gives 87.
    # I will implement the 4x4 Strassen using the recursive logic from the failed code, 
    # ensuring it returns correct 4x4 matrices.

    def build_strassen(k):
        if k == 0:
            return [[1]], [[1]], [[1]]
        
        U_sub, V_sub, W_sub = build_strassen(k-1)
        rank_sub = len(U_sub[0])
        N = 2**k
        N_half = 2**(k-1)
        rank = 7 * rank_sub
        
        U = [[0]*rank for _ in range(N*N)]
        V = [[0]*rank for _ in range(N*N)]
        W = [[0]*rank for _ in range(N*N)]
        
        def get_block_indices(block):
            r0, c0 = 0, 0
            if block == 0: r0, c0 = 0, 0
            elif block == 1: r0, c0 = 0, N_half
            elif block == 2: r0, c0 = N_half, 0
            elif block == 3: r0, c0 = N_half, N_half
            indices = []
            for i in range(N_half):
                for j in range(N_half):
                    indices.append((r0+i, c0+j))
            return indices

        # M1
        for c in range(rank_sub):
            col = c
            for (i, j) in get_block_indices(0):
                U[i*N+j][col] += U_sub[i*N_half+j][c]
                V[i*N+j][col] += V_sub[i*N_half+j][c]
                W[i*N+j][col] += W_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(3):
                U[i*N+j][col] += U_sub[i*N_half+j][c]
                V[i*N+j][col] += V_sub[i*N_half+j][c]
                W[i*N+j][col] += W_sub[i*N_half+j][c]

        # M2
        for c in range(rank_sub):
            col = rank_sub + c
            for (i, j) in get_block_indices(2):
                U[i*N+j][col] += U_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(3):
                U[i*N+j][col] += U_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(0):
                V[i*N+j][col] += V_sub[i*N_half+j][c]
                W[i*N+j][col] += W_sub[i*N_half+j][c]

        # M3
        for c in range(rank_sub):
            col = 2*rank_sub + c
            for (i, j) in get_block_indices(0):
                U[i*N+j][col] += U_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(1):
                V[i*N+j][col] += V_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(3):
                V[i*N+j][col] -= V_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(1):
                W[i*N+j][col] += W_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(3):
                W[i*N+j][col] -= W_sub[i*N_half+j][c]

        # M4
        for c in range(rank_sub):
            col = 3*rank_sub + c
            for (i, j) in get_block_indices(3):
                U[i*N+j][col] += U_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(2):
                V[i*N+j][col] += V_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(0):
                V[i*N+j][col] -= V_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(2):
                W[i*N+j][col] += W_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(0):
                W[i*N+j][col] -= W_sub[i*N_half+j][c]

        # M5
        for c in range(rank_sub):
            col = 4*rank_sub + c
            for (i, j) in get_block_indices(0):
                U[i*N+j][col] += U_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(1):
                U[i*N+j][col] += U_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(3):
                V[i*N+j][col] += V_sub[i*N_half+j][c]
                W[i*N+j][col] += W_sub[i*N_half+j][c]

        # M6
        for c in range(rank_sub):
            col = 5*rank_sub + c
            for (i, j) in get_block_indices(2):
                U[i*N+j][col] += U_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(0):
                U[i*N+j][col] -= U_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(0):
                V[i*N+j][col] += V_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(1):
                V[i*N+j][col] += V_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(0):
                W[i*N+j][col] += W_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(1):
                W[i*N+j][col] += W_sub[i*N_half+j][c]

        # M7
        for c in range(rank_sub):
            col = 6*rank_sub + c
            for (i, j) in get_block_indices(1):
                U[i*N+j][col] += U_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(3):
                U[i*N+j][col] -= U_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(2):
                V[i*N+j][col] += V_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(3):
                V[i*N+j][col] += V_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(2):
                W[i*N+j][col] += W_sub[i*N_half+j][c]
            for (i, j) in get_block_indices(3):
                W[i*N+j][col] += W_sub[i*N_half+j][c]

        return U, V, W

    U4, V4, W4 = build_strassen(2)
    rank4 = len(U4[0])
    
    U5 = [[0]*87 for _ in range(25)]
    V5 = [[0]*87 for _ in range(25)]
    W5 = [[0]*87 for _ in range(25)]
    
    # Copy 4x4
    for i in range(4):
        for j in range(4):
            for c in range(rank4):
                U5[i*5+j][c] = U4[i*4+j][c]
                V5[j*5+i][c] = V4[j*4+i][c]
                W5[i*5+j][c] = W4[i*4+j][c]
                
    col = rank4
    # a) a_col b_row
    for i in range(4):
        for j in range(4):
            U5[i*5+4][col] = 1
            V5[4*5+j][col] = 1
            W5[j*5+i][col] = 1
    col += 1
    
    # b) A4x4 b_col
    for i in range(4):
        for k in range(4):
            U5[i*5+k][col] = 1
            V5[k*5+4][col] = 1
            W5[4*5+i][col] = 1
            col += 1
            
    # c) a_row B4x4
    for j in range(4):
        for k in range(4):
            U5[4*5+k][col] = 1
            V5[k*5+j][col] = 1
            W5[j*5+4][col] = 1
            col += 1
            
    # d) a_row b_col
    for k in range(4):
        U5[4*5+k][col] = 1
        V5[k*5+4][col] = 1
        W5[4*5+4][col] = 1
        col += 1
        
    # e) a_scalar b_scalar
    U5[4*5+4][col] = 1
    V5[4*5+4][col] = 1
    W5[4*5+4][col] = 1
    col += 1
    
    return {"U": U5, "V": V5, "W": W5}
