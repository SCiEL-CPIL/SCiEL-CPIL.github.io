import time

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Construct a rank-55 exact tensor decomposition for 2x3 @ 3x10 matrix multiplication.
    
    Strategy: Split B into 5 blocks of width 2. For each block, split A into left 2 columns 
    and right 1 column. Use Strassen's 7-term algorithm for the 2x2 @ 2x2 multiplication 
    (A_left @ B_block) and 4 trivial terms for A_right @ B_block.
    Total rank = 5 blocks * (7 + 4) = 55.
    
    Trace convention: W[k*n + i, r] where C = A @ B, n=2, m=3, p=10.
    A index: i*m + j, B index: j*p + k, C index: k*n + i
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    num_blocks = p // 2   # 5 blocks of width 2
    rank = num_blocks * 11  # 55
    
    U = [[0] * rank for _ in range(n * m)]  # (6, 55)
    V = [[0] * rank for _ in range(m * p)]  # (30, 55)
    W = [[0] * rank for _ in range(p * n)]  # (20, 55)
    
    for block in range(num_blocks):
        base_col = block * 11
        b_start = block * 2  # starting column index in B
        
        # A indices (i*m+j): A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5
        # B indices (j*p+k): B[j, b_start], B[j, b_start+1], B[j, b_start+2] etc.
        # C indices (k*n+i): C[i, b_start], C[i, b_start+1], C[i, b_start+2] etc.
        
        # === Strassen 7 terms for A_left (cols 0,1) @ B_block (rows 0,1; cols b_start, b_start+1) ===
        
        # Term 1: M1 = (A[0,0]+A[1,1]) * (B[0,b_start]+B[1,b_start+1])
        # Contributes: C[0,b_start] += M1, C[1,b_start+1] += M1
        col = base_col
        U[0][col] = 1   # A[0,0]
        U[4][col] = 1   # A[1,1]
        V[0*p + b_start][col] = 1       # B[0, b_start]
        V[1*p + b_start + 1][col] = 1   # B[1, b_start+1]
        W[b_start*n + 0][col] = 1       # C[0, b_start]
        W[(b_start+1)*n + 1][col] = 1   # C[1, b_start+1]
        
        # Term 2: M2 = (A[1,0]+A[1,1]) * B[0,b_start]
        # Contributes: C[0,b_start+1] += M2, C[1,b_start] += M2
        col = base_col + 1
        U[3][col] = 1   # A[1,0]
        U[4][col] = 1   # A[1,1]
        V[0*p + b_start][col] = 1       # B[0, b_start]
        W[(b_start+1)*n + 0][col] = 1   # C[0, b_start+1]
        W[b_start*n + 1][col] = 1       # C[1, b_start]
        
        # Term 3: M3 = A[0,0] * (B[1,b_start] - B[1,b_start+1])
        # Contributes: C[0,b_start+1] += M3, C[1,b_start] += M3
        col = base_col + 2
        U[0][col] = 1   # A[0,0]
        V[1*p + b_start][col] = 1       # B[1, b_start]
        V[1*p + b_start + 1][col] = -1  # B[1, b_start+1]
        W[(b_start+1)*n + 0][col] = 1   # C[0, b_start+1]
        W[b_start*n + 1][col] = 1       # C[1, b_start]
        
        # Term 4: M4 = A[1,1] * (B[1,b_start] - B[0,b_start+1])
        # Contributes: C[0,b_start] += M4, C[1,b_start+1] += M4
        col = base_col + 3
        U[4][col] = 1   # A[1,1]
        V[1*p + b_start][col] = 1       # B[1, b_start]
        V[0*p + b_start + 1][col] = -1  # B[0, b_start+1]
        W[b_start*n + 0][col] = 1       # C[0, b_start]
        W[(b_start+1)*n + 1][col] = 1   # C[1, b_start+1]
        
        # Term 5: M5 = (A[0,0]+A[0,1]) * B[1,b_start+1]
        # Contributes: C[0,b_start] -= M5
        col = base_col + 4
        U[0][col] = 1   # A[0,0]
        U[1][col] = 1   # A[0,1]
        V[1*p + b_start + 1][col] = 1   # B[1, b_start+1]
        W[b_start*n + 0][col] = -1      # C[0, b_start]
        
        # Term 6: M6 = (A[1,0]-A[0,0]) * (B[0,b_start]+B[0,b_start+1])
        # Contributes: C[1,b_start+1] += M6
        col = base_col + 5
        U[3][col] = 1    # A[1,0]
        U[0][col] = -1   # A[0,0]
        V[0*p + b_start][col] = 1       # B[0, b_start]
        V[0*p + b_start + 1][col] = 1   # B[0, b_start+1]
        W[(b_start+1)*n + 1][col] = 1   # C[1, b_start+1]
        
        # Term 7: M7 = (A[0,1]-A[1,1]) * (B[1,b_start]+B[1,b_start+1])
        # Contributes: C[0,b_start] += M7
        col = base_col + 6
        U[1][col] = 1    # A[0,1]
        U[4][col] = -1   # A[1,1]
        V[1*p + b_start][col] = 1       # B[1, b_start]
        V[1*p + b_start + 1][col] = 1   # B[1, b_start+1]
        W[b_start*n + 0][col] = 1       # C[0, b_start]
        
        # === 4 trivial terms for A_right (col 2) @ B_block (row 2; cols b_start, b_start+1) ===
        
        # Term 8: A[0,2] * B[2,b_start] -> C[0,b_start] += 
        col = base_col + 7
        U[2][col] = 1   # A[0,2]
        V[2*p + b_start][col] = 1       # B[2, b_start]
        W[b_start*n + 0][col] = 1       # C[0, b_start]
        
        # Term 9: A[0,2] * B[2,b_start+1] -> C[0,b_start+1] +=
        col = base_col + 8
        U[2][col] = 1   # A[0,2]
        V[2*p + b_start + 1][col] = 1   # B[2, b_start+1]
        W[(b_start+1)*n + 0][col] = 1   # C[0, b_start+1]
        
        # Term 10: A[1,2] * B[2,b_start] -> C[1,b_start] +=
        col = base_col + 9
        U[5][col] = 1   # A[1,2]
        V[2*p + b_start][col] = 1       # B[2, b_start]
        W[b_start*n + 1][col] = 1       # C[1, b_start]
        
        # Term 11: A[1,2] * B[2,b_start+1] -> C[1,b_start+1] +=
        col = base_col + 10
        U[5][col] = 1   # A[1,2]
        V[2*p + b_start + 1][col] = 1   # B[2, b_start+1]
        W[(b_start+1)*n + 1][col] = 1   # C[1, b_start+1]
    
    return {"U": U, "V": V, "W": W}
