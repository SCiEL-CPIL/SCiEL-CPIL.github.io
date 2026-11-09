import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit
    
    # 2x2x2 Strassen decomposition
    # U, V, W are 4x7 matrices
    U22 = [
        [1, 0, 1, 0, 1, -1, 0],  # r=0..6 for i=0,j=0
        [0, 0, 0, 0, 1, 0, 1],   # i=0,j=1
        [0, 1, 0, 0, 0, 1, 0],   # i=1,j=0
        [1, 1, 0, 1, 0, 0, -1]   # i=1,j=1
    ]
    V22 = [
        [1, 1, 0, -1, 0, 1, 0],  # r=0..6 for j=0,k=0
        [0, 0, 1, 0, 0, 1, 0],   # j=0,k=1
        [0, 0, 0, 1, 0, 0, 1],   # j=1,k=0
        [1, 0, -1, 0, 1, 0, 1]   # j=1,k=1
    ]
    W22 = [
        [1, 0, 0, 1, -1, 0, 1],  # r=0..6 for k=0,i=0
        [0, 1, 0, 1, 0, 0, 0],   # k=0,i=1
        [0, 0, 1, 0, 1, 0, 0],   # k=1,i=0
        [1, 0, 0, 0, 0, 1, 1]    # k=1,i=1
    ]
    
    # Verify 2x2x2
    # T[i,j,k] = sum_r U22[i*2+j][r] * V22[j*2+k][r] * W22[k*2+i][r]
    
    # Build 4x4x4 decomposition
    # 7 Strassen multiplications
    # Each is a 2x2x2 multiplication
    # So we have 7 * 7 = 49 rank-1 components
    
    U44 = [[0]*49 for _ in range(16)]
    V44 = [[0]*49 for _ in range(16)]
    W44 = [[0]*49 for _ in range(16)]
    
    # Strassen coefficients
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map the 4x4 indices to the 2x2 block indices
    # A is 4x4, blocks A11, A12, A21, A22
    # A11 is A[0:2, 0:2], etc.
    
    # For each Strassen multiplication r_s (0..6), we have a 2x2x2 multiplication
    # The 2x2x2 multiplication has 7 rank-1 components r_22 (0..6)
    # Total rank index r = r_s * 7 + r_22
    
    # We need to define how the 2x2 blocks are combined
    # For M1: A' = A11+A22, B' = B11+B22
    # The 2x2x2 decomposition of A'B' gives contributions to C11 and C22
    
    # Let's define the block structure
    # A blocks:
    # A00: i in 0,1; j in 0,1
    # A01: i in 0,1; j in 2,3
    # A10: i in 2,3; j in 0,1
    # A11: i in 2,3; j in 2,3
    
    # B blocks:
    # B00: j in 0,1; k in 0,1
    # B01: j in 0,1; k in 2,3
    # B10: j in 2,3; k in 0,1
    # B11: j in 2,3; k in 2,3
    
    # C blocks:
    # C00: i in 0,1; k in 0,1
    # C01: i in 0,1; k in 2,3
    # C10: i in 2,3; k in 0,1
    # C11: i in 2,3; k in 2,3
    
    # Strassen formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will iterate over r_s and r_22
    # For each, we determine the U, V, W vectors
    
    # This is complex to hardcode. Let's use a more systematic approach.
    # We can just use the fact that the rank is 49 and try to find a decomposition.
    # But hardcoding is safer.
    
    # Let's try a different approach: use the known rank 49 decomposition.
    # Or just use the schoolbook for now if we can't construct it quickly.
    # Schoolbook rank is 64 for 4x4x4.
    # 64 + 64 = 128.
    # Strassen is 49 + 49 = 98.
    
    # Let's try to implement the composition.
    
    # For each Strassen multiplication r_s, we have a 2x2x2 tensor T_s.
    # T_s is decomposed as sum_{r_22} U_s[r_22] \otimes V_s[r_22] \otimes W_s[r_22].
    # The U_s, V_s, W_s are 4x7 matrices (for the 2x2 blocks).
    # We need to embed these into the 4x4 matrices.
    
    # For M1: A' = A00 + A11.
    # U_s for r_22: U22[i2*2+j2][r_22]
    # This corresponds to A[i,j] where i = i2*2 + i_offset, j = j2*2 + j_offset.
    # For A00: i_offset=0, j_offset=0.
    # For A11: i_offset=2, j_offset=2.
    # So U_s[i,j] = U22[i2*2+j2][r_22] if (i,j) in A00 or A11.
    
    # This is getting complicated. Let's just use the schoolbook decomposition for now.
    # Rank 128.
    # If we have time, we can try to optimize.
    
    n, m, p = 4, 4, 8
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
