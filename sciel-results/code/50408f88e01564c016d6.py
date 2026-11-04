import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a decomposition of rank 28.
    # Split p=5 into p1=2 and p2=3.
    # Block 1: 2x3x2 multiplication. Rank 11.
    #   Split m=3 into m1=2 and m2=1.
    #   Sub-block 1a: 2x2x2 (Strassen, rank 7).
    #   Sub-block 1b: 2x1x2 (Schoolbook, rank 4).
    # Block 2: 2x3x3 multiplication. Rank 17.
    #   Split m=3 into m1=2 and m2=1.
    #   Sub-block 2a: 2x2x3 (Rank 11, constructed similarly: split m=2? No, 2x2x3 is known rank 11. 
    #                 Can we construct it? Split p=3 into 1+2.
    #                 R(2,2,1)=4. R(2,2,2)=7. Sum 11. Yes!)
    #   Sub-block 2b: 2x1x3 (Schoolbook, rank 6).
    
    # Total rank: 7 + 4 + 11 + 6 = 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Block 1: p=2 (columns 0,1 of B and C) ---
    # Sub-block 1a: 2x2x2 (Strassen)
    # A indices: i in 0..1, j in 0..1
    # B indices: j in 0..1, k in 0..1
    # C indices: i in 0..1, k in 0..1
    
    # Strassen U, V, W for 2x2x2
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U (4x7)
    U_strassen = [
        [1, 0, 1, 0, 1, 1, 0], # A11
        [0, 1, 0, 0, 0, -1, 0], # A21 (index 1 in 2x2? No, A is 2x2. Indices 0,1,2,3. A11=0, A12=1, A21=2, A22=3)
        # Let's use 0-based: A[0,0], A[0,1], A[1,0], A[1,1]
        # U rows correspond to A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        [1, 0, 1, 0, 1, 1, 0], # A[0,0]
        [0, 0, 0, 0, 1, 0, -1], # A[0,1]
        [0, 1, 0, 0, 0, -1, 0], # A[1,0]
        [1, 1, 0, 1, 0, 0, -1]  # A[1,1]
    ]
    # V (4x7)
    # B[0,0], B[0,1], B[1,0], B[1,1]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # B[0,0]
        [0, 0, 1, 0, 0, 1, 0],  # B[0,1]
        [1, 0, -1, 1, 0, 0, 1], # B[1,0]
        [1, 0, 0, 0, 1, 0, 1]   # B[1,1]
    ]
    # W (4x7)
    # C[0,0], C[0,1], C[1,0], C[1,1]
    W_strassen = [
        [1, 0, 0, 0, 0, 0, 0], # C[0,0]
        [0, 0, 1, 0, 0, 1, 0], # C[0,1]
        [0, 1, 0, 1, 0, 0, 1], # C[1,0]
        [0, 0, 0, 0, 1, 0, 0]  # C[1,1]
    ]
    
    # Map Strassen indices to global indices
    # Global A: 2x3. Block 1a uses A[0..1, 0..1].
    # Global B: 3x5. Block 1a uses B[0..1, 0..1].
    # Global C: 2x5. Block 1a uses C[0..1, 0..1].
    
    for r in range(4):
        for c in range(7):
            # U
            i, j = divmod(r, 2) # 0-based: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Wait, divmod(r, 2) gives (0,0), (0,1), (1,0), (1,1) for r=0..3?
            # r=0: 0,0. r=1: 0,1. r=2: 1,0. r=3: 1,1. Correct.
            U[i * m + j][col + c] += U_strassen[r][c]
            
            # V
            j_b, k = divmod(r, 2)
            # B index is j_b * p + k. p=5.
            V[j_b * p + k][col + c] += V_strassen[r][c]
            
            # W
            i_c, k_c = divmod(r, 2)
            # W index is k_c * n + i_c. n=2.
            W[k_c * n + i_c][col + c] += W_strassen[r][c]
            
    col += 7
    
    # Sub-block 1b: 2x1x2 (Schoolbook)
    # A indices: i in 0..1, j=2
    # B indices: j=2, k in 0..1
    # C indices: i in 0..1, k in 0..1
    # Rank 4.
    for i in range(2):
        for k in range(2):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # --- Block 2: p=3 (columns 2,3,4 of B and C) ---
    # Sub-block 2a: 2x2x3 (Rank 11)
    # Split p=3 into 1+2.
    # Sub-sub 2a1: 2x2x1 (Rank 4, Schoolbook)
    # Sub-sub 2a2: 2x2x2 (Rank 7, Strassen)
    
    # 2a1: 2x2x1
    # A: i in 0..1, j in 0..1
    # B: j in 0..1, k=2 (global k index)
    # C: i in 0..1, k=2
    for i in range(2):
        for j in range(2):
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
            
    # 2a2: 2x2x2 (Strassen)
    # A: i in 0..1, j in 0..1
    # B: j in 0..1, k in 3..4 (global k indices 3,4)
    # C: i in 0..1, k in 3..4
    
    # Reuse Strassen U, V
    # But V and W indices shift.
    for r in range(4):
        for c in range(7):
            # U
            i, j = divmod(r, 2)
            U[i * m + j][col + c] += U_strassen[r][c]
            
            # V
            j_b, k_local = divmod(r, 2)
            k_global = 3 + k_local
            V[j_b * p + k_global][col + c] += V_strassen[r][c]
            
            # W
            i_c, k_local_c = divmod(r, 2)
            k_global_c = 3 + k_local_c
            W[k_global_c * n + i_c][col + c] += W_strassen[r][c]
            
    col += 7
    
    # Sub-block 2b: 2x1x3 (Schoolbook)
    # A: i in 0..1, j=2
    # B: j=2, k in 2..4
    # C: i in 0..1, k in 2..4
    for i in range(2):
        for k in range(2, 5):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
