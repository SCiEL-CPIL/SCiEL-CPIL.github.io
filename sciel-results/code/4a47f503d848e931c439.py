import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    
    # We will use 5 blocks of 2 columns for B
    num_blocks = p // 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)] # 6 x R
    V = [[0] * total_rank for _ in range(m * p)] # 30 x R
    W = [[0] * total_rank for _ in range(p * n)] # 20 x R
    
    # Strassen factors for 2x2x2
    # U indices: 0,1,2,3 for A11, A12, A21, A22
    # V indices: 0,1,2,3 for B11, B12, B21, B22
    # W indices: 0,1,2,3 for C11, C12, C21, C22
    
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    strassen_U = [
        [1, 0, 1, 0, 1, -1, -1], # A11
        [0, 1, 0, 0, 1, 0, 1],   # A12
        [0, 0, 0, 1, 0, 1, 0],   # A21
        [1, 1, 0, 1, 0, 0, -1]   # A22
    ]
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0],  # B11
        [0, 0, 1, 0, 0, 1, 0],   # B12
        [0, 0, 0, 1, 0, 0, 1],   # B21
        [1, 0, -1, 0, 1, 0, 1]   # B22
    ]
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],  # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, -1, 1, 0, 0, 1, 0]   # C22
    ]
    
    # Rank 4 factors for 2x1x2
    # A is 2x1 (A11, A21)
    # B is 1x2 (B11, B12)
    # C is 2x2 (C11, C12, C21, C22)
    # C11 = A11 B11
    # C12 = A11 B12
    # C21 = A21 B11
    # C22 = A21 B12
    
    outer_U = [
        [1, 1, 0, 0], # A11
        [0, 0, 1, 1]  # A21
    ]
    outer_V = [
        [1, 0, 1, 0], # B11
        [0, 1, 0, 1]  # B12
    ]
    outer_W = [
        [1, 0, 0, 0], # C11
        [0, 1, 0, 0], # C12
        [0, 0, 1, 0], # C21
        [0, 0, 0, 1]  # C22
    ]
    
    col_offset = 0
    for j in range(num_blocks):
        # Block j uses columns 2j, 2j+1 of B and C
        c_start = 2 * j
        
        # --- Part 1: A2 (cols 1,2 of A) * B2 (rows 1,2 of B, cols c_start, c_start+1) ---
        # A2 is 2x2. Indices in A: 
        # A is 2x3. 
        # A2 corresponds to A[0,1], A[0,2], A[1,1], A[1,2]
        # Flattened A indices:
        # A[0,0]=0, A[0,1]=1, A[0,2]=2
        # A[1,0]=3, A[1,1]=4, A[1,2]=5
        # So A2 uses indices 1, 2, 4, 5.
        
        # B2 is 2x2. Indices in B:
        # B is 3x10.
        # B2 corresponds to B[1, c_start], B[1, c_start+1], B[2, c_start], B[2, c_start+1]
        # Flattened B indices:
        # B[i, k] = i * 10 + k
        # B[1, c_start] = 10 + c_start
        # B[1, c_start+1] = 10 + c_start + 1
        # B[2, c_start] = 20 + c_start
        # B[2, c_start+1] = 20 + c_start + 1
        
        # C part:
        # C is 2x10.
        # C corresponds to C[0, c_start], C[0, c_start+1], C[1, c_start], C[1, c_start+1]
        # Flattened C indices (W is p*n = 10*2 = 20 rows? No, W[k*n+i, r])
        # W index is k*n + i.
        # C[0, c_start] -> k=c_start, i=0 -> c_start*2 + 0
        # C[0, c_start+1] -> k=c_start+1, i=0 -> (c_start+1)*2 + 0
        # C[1, c_start] -> k=c_start, i=1 -> c_start*2 + 1
        # C[1, c_start+1] -> k=c_start+1, i=1 -> (c_start+1)*2 + 1
        
        # Map Strassen local indices to global
        # Local A2: 0->A[0,1], 1->A[0,2], 2->A[1,1], 3->A[1,2]
        # Global A indices: 1, 2, 4, 5
        
        # Local B2: 0->B[1, c], 1->B[1, c+1], 2->B[2, c], 3->B[2, c+1]
        # Global B indices: 10+c, 10+c+1, 20+c, 20+c+1
        
        # Local C: 0->C[0, c], 1->C[0, c+1], 2->C[1, c], 3->C[1, c+1]
        # Global W indices: 2*c, 2*c+2, 2*c+1, 2*c+3
        
        for r in range(7):
            col = col_offset + r
            # U
            U[1][col] += strassen_U[0][r]
            U[2][col] += strassen_U[1][r]
            U[4][col] += strassen_U[2][r]
            U[5][col] += strassen_U[3][r]
            
            # V
            V[10 + c_start][col] += strassen_V[0][r]
            V[10 + c_start + 1][col] += strassen_V[1][r]
            V[20 + c_start][col] += strassen_V[2][r]
            V[20 + c_start + 1][col] += strassen_V[3][r]
            
            # W
            W[2 * c_start][col] += strassen_W[0][r]
            W[2 * c_start + 2][col] += strassen_W[1][r]
            W[2 * c_start + 1][col] += strassen_W[2][r]
            W[2 * c_start + 3][col] += strassen_W[3][r]
            
        col_offset += 7
        
        # --- Part 2: A1 (col 0 of A) * B1 (row 0 of B, cols c_start, c_start+1) ---
        # A1 is 2x1. Indices in A: 0, 3.
        # B1 is 1x2. Indices in B: 0 + c_start, 0 + c_start + 1.
        # C part is same as above.
        
        # Local A1: 0->A[0,0], 1->A[1,0]
        # Global A indices: 0, 3
        
        # Local B1: 0->B[0, c], 1->B[0, c+1]
        # Global B indices: c, c+1
        
        # Local C: same mapping
        
        for r in range(4):
            col = col_offset + r
            # U
            U[0][col] += outer_U[0][r]
            U[3][col] += outer_U[1][r]
            
            # V
            V[c_start][col] += outer_V[0][r]
            V[c_start + 1][col] += outer_V[1][r]
            
            # W
            W[2 * c_start][col] += outer_W[0][r]
            W[2 * c_start + 2][col] += outer_W[1][r]
            W[2 * c_start + 1][col] += outer_W[2][r]
            W[2 * c_start + 3][col] += outer_W[3][r]
            
        col_offset += 4
        
    return {"U": U, "V": V, "W": W}
