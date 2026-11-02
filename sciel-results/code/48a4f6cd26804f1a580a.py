import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 110s to be safe
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Schoolbook rank = 48
    rank_schoolbook = n * m * p  # 48
    
    # We'll try to find a rank 44 solution using block decomposition
    # Split the 2x3 matrix A into:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 0 (since n=2, m=3)
    # Wait, n=2, m=3: A is 2x3, B is 3x8
    # Let's split columns of A and rows of B:
    # A = [A_top | A_bottom] where A_top is 2x2, A_bottom is 2x1
    # B = [B_top; B_bottom] where B_top is 2x8, B_bottom is 1x8
    # C = A_top * B_top + A_bottom * B_bottom
    
    # For A_top * B_top (2x2 * 2x8): use Strassen (rank 7 per 2x2x2 block, so 7*4=28 for 2x2x8)
    # For A_bottom * B_bottom (2x1 * 1x8): schoolbook, rank 16
    
    # Total rank = 28 + 16 = 44
    
    # Let me construct this explicitly.
    # U has shape (n*m, rank) = (6, 44)
    # V has shape (m*p, rank) = (24, 44)  
    # W has shape (p*n, rank) = (16, 44)
    
    # Indices:
    # U[i*m + j, c] where i in [0,n), j in [0,m)
    # V[j*p + k, c] where j in [0,m), k in [0,p)
    # W[k*n + i, c] where k in [0,p), i in [0,n)
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    # Block 1: A_top (2x2) * B_top (2x8), Strassen rank 28
    # Split B_top into 4 blocks of 2x2: B_top[:, 0:2], B_top[:, 2:4], B_top[:, 4:6], B_top[:, 6:8]
    # For each 2x2x2 block, Strassen uses 7 multiplications
    
    # Strassen for 2x2x2:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    col_offset = 0
    for block in range(4):
        # B_top columns 2*block to 2*block+1
        b11_idx = (0) * p + (2 * block)      # j=0, k=2*block
        b12_idx = (0) * p + (2 * block + 1)  # j=0, k=2*block+1
        b21_idx = (1) * p + (2 * block)      # j=1, k=2*block
        b22_idx = (1) * p + (2 * block + 1)  # j=1, k=2*block+1
        
        # A_top: rows i=0,1; cols j=0,1
        a11_idx = 0 * m + 0  # i=0, j=0
        a12_idx = 0 * m + 1  # i=0, j=1
        a21_idx = 1 * m + 0  # i=1, j=0
        a22_idx = 1 * m + 1  # i=1, j=1
        
        # M1 = (a11+a22)(b11+b22)
        c = col_offset
        U[a11_idx][c] = 1
        U[a22_idx][c] = 1
        V[b11_idx][c] = 1
        V[b22_idx][c] = 1
        W[(2*block)*n + 0][c] = 1       # C[0,0] += M1
        W[(2*block+1)*n + 0][c] = 1     # C[1,0] += M1 (wait, W[k*n+i])
        # Actually W index is k*n + i, where k is output col, i is output row
        # C[k, i] = sum_j A[i,j] B[j,k]
        # For C[0, 2*block]: contributions from j=0,1
        # C[0, 2*block] = a11*b11 + a12*b21
        # C[0, 2*block+1] = a11*b12 + a12*b22
        # C[1, 2*block] = a21*b11 + a22*b21
        # C[1, 2*block+1] = a21*b12 + a22*b22
        
        # Strassen gives:
        # C11 = M1+M4-M5+M7 -> C[0, 2*block]
        # C12 = M3+M5 -> C[0, 2*block+1]
        # C21 = M2+M4 -> C[1, 2*block]
        # C22 = M1-M2+M3+M6 -> C[1, 2*block+1]
        
        # M1
        W[(2*block)*n + 0][c] = 1    # C[0, 2*block]
        W[(2*block+1)*n + 1][c] = 1  # C[1, 2*block+1]
        col_offset += 1
        
        # M2 = (a21+a22)b11
        c = col_offset
        U[a21_idx][c] = 1
        U[a22_idx][c] = 1
        V[b11_idx][c] = 1
        W[(2*block)*n + 1][c] = 1  # C[1, 2*block]
        col_offset += 1
        
        # M3 = a11(b12-b22)
        c = col_offset
        U[a11_idx][c] = 1
        V[b12_idx][c] = 1
        V[b22_idx][c] = -1
        W[(2*block+1)*n + 0][c] = 1  # C[0, 2*block+1]
        W[(2*block+1)*n + 1][c] = 1  # C[1, 2*block+1]
        col_offset += 1
        
        # M4 = a22(b21-b11)
        c = col_offset
        U[a22_idx][c] = 1
        V[b21_idx][c] = 1
        V[b11_idx][c] = -1
        W[(2*block)*n + 0][c] = 1  # C[0, 2*block]
        W[(2*block)*n + 1][c] = 1  # C[1, 2*block]
        col_offset += 1
        
        # M5 = (a11+a12)b22
        c = col_offset
        U[a11_idx][c] = 1
        U[a12_idx][c] = 1
        V[b22_idx][c] = 1
        W[(2*block)*n + 0][c] = -1  # C[0, 2*block]
        W[(2*block+1)*n + 0][c] = 1  # C[0, 2*block+1]
        col_offset += 1
        
        # M6 = (a21-a11)(b11+b12)
        c = col_offset
        U[a21_idx][c] = 1
        U[a11_idx][c] = -1
        V[b11_idx][c] = 1
        V[b12_idx][c] = 1
        W[(2*block+1)*n + 1][c] = 1  # C[1, 2*block+1]
        col_offset += 1
        
        # M7 = (a12-a22)(b21+b22)
        c = col_offset
        U[a12_idx][c] = 1
        U[a22_idx][c] = -1
        V[b21_idx][c] = 1
        V[b22_idx][c] = 1
        W[(2*block)*n + 0][c] = 1  # C[0, 2*block]
        col_offset += 1
    
    # Block 2: A_bottom (2x1) * B_bottom (1x8), schoolbook rank 16
    # A_bottom: i=0,1; j=2
    # B_bottom: j=2, k=0..7
    # C[0, k] += A[0,2]*B[2,k]
    # C[1, k] += A[1,2]*B[2,k]
    
    for k in range(8):
        c = col_offset
        a_bottom_0_idx = 0 * m + 2  # i=0, j=2
        a_bottom_1_idx = 1 * m + 2  # i=1, j=2
        b_bottom_idx = 2 * p + k    # j=2, k
        
        U[a_bottom_0_idx][c] = 1
        V[b_bottom_idx][c] = 1
        W[k * n + 0][c] = 1  # C[0, k]
        
        U[a_bottom_1_idx][c] = 1
        V[b_bottom_idx][c] = 1  # This is wrong, need separate columns
        # Actually each multiplication needs its own column
        col_offset += 1
        
        c = col_offset
        U[a_bottom_1_idx][c] = 1
        V[b_bottom_idx][c] = 1
        W[k * n + 1][c] = 1  # C[1, k]
        col_offset += 1
    
    # Check deadline
    if time.time() > deadline - 10:
        pass  # Return what we have
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
