"""Matrix multiplication 2x3x10 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split p=10 into 5 blocks of size 2
    # Each block is a 2x3x2 multiplication
    # Rank of 2x3x2 is 11 (7 for 2x2x2 part + 4 for 2x1x2 part)
    # Total rank = 5 * 11 = 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We define the 11 terms for one 2x3x2 block
    # A is 2x3, B is 3x2, C is 2x2
    # A = [A00 A01 A02; A10 A11 A12]
    # B = [B00 B01; B10 B11; B20 B21]
    # C = A * B
    
    # We use Strassen for the 2x2 part (A_L * B_T) and outer product for A_R * B_B
    # A_L = [A00 A01; A10 A11]
    # B_T = [B00 B01; B10 B11]
    # A_R = [A02; A12]
    # B_B = [B20 B21]
    
    # Strassen terms for A_L * B_T (indices 0-6)
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # Outer product terms for A_R * B_B (indices 7-10)
    # M7 = A02 * B20
    # M8 = A02 * B21
    # M9 = A12 * B20
    # M10 = A12 * B21
    
    # We need to map these to the full U, V, W matrices
    # For each block b (0 to 4):
    # The columns of B are 2b, 2b+1
    # The rows of B are 0, 1, 2
    # So B indices are:
    # B[0, 2b] = B00
    # B[0, 2b+1] = B01
    # B[1, 2b] = B10
    # B[1, 2b+1] = B11
    # B[2, 2b] = B20
    # B[2, 2b+1] = B21
    
    # A indices are global:
    # A[0,0]=A00, A[0,1]=A01, A[0,2]=A02
    # A[1,0]=A10, A[1,1]=A11, A[1,2]=A12
    
    # C indices for block b:
    # C[0, 2b]=C00, C[0, 2b+1]=C01
    # C[1, 2b]=C10, C[1, 2b+1]=C11
    # W index is k*n + i, so:
    # W[2b*2 + 0] = C00
    # W[2b*2 + 1] = C01
    # W[(2b+1)*2 + 0] = C10
    # W[(2b+1)*2 + 1] = C11
    
    # Define the 11 basis vectors for one block
    # U_part (6x11), V_part (6x11), W_part (4x11)
    
    block_U = [
        [1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0], # A00
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # A01
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], # A02
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1], # A11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Placeholder, A12 is row 5? 
        # Wait, A is 2x3. Rows 0,1.
        # A[0,0], A[0,1], A[0,2] are indices 0,1,2 in U?
        # U is (n*m) x rank. n=2, m=3.
        # U[0] = A00, U[1] = A01, U[2] = A02
        # U[3] = A10, U[4] = A11, U[5] = A12
    ]
    
    # Correct U_part
    block_U = [
        [1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0], # U[0] A00
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # U[1] A01
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], # U[2] A02
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0], # U[3] A10
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1], # U[4] A11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # U[5] A12? 
    ]
    # Wait, A12 is involved in M7, M8, M9, M10.
    # M7 = A02 B20
    # M8 = A02 B21
    # M9 = A12 B20
    # M10 = A12 B21
    # So U[5] (A12) should have 1s in cols 9, 10.
    block_U[5] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    
    # V_part (6x11)
    # V indices:
    # V[0] = B00, V[1] = B01
    # V[2] = B10, V[3] = B11
    # V[4] = B20, V[5] = B21
    block_V = [
        [1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0], # B00
        [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0], # B01
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0], # B10
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # B11
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # B20
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  # B21
    ]
    # Check B20, B21
    # M7 = A02 B20 -> V[4] col 7 = 1
    # M8 = A02 B21 -> V[5] col 8 = 1
    # M9 = A12 B20 -> V[4] col 9 = 1
    # M10 = A12 B21 -> V[5] col 10 = 1
    block_V[4] = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]
    block_V[5] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
    
    # W_part (4x11)
    # W indices for this block:
    # W[0] = C00
    # W[1] = C01
    # W[2] = C10
    # W[3] = C11
    
    # C00 = M0 - M1 - M3 + M5 + M7
    # C01 = M2 + M4 + M6 + M8
    # C10 = M1 + M3 + M5 + M9
    # C11 = M0 - M2 - M4 - M6 + M10
    
    block_W = [
        [1, -1, 0, -1, 0, 1, 0, 1, 0, 0, 0], # C00
        [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0],   # C10
        [1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 1] # C11
    ]
    
    # Fill U, V, W
    for b in range(5):
        offset = b * 11
        for i in range(6):
            for k in range(11):
                U[i][offset + k] = block_U[i][k]
        
        # V indices for block b
        # B rows 0,1,2; cols 2b, 2b+1
        # V[0, 2b] = B00
        # V[0, 2b+1] = B01
        # V[1, 2b] = B10
        # V[1, 2b+1] = B11
        # V[2, 2b] = B20
        # V[2, 2b+1] = B21
        # V is (m*p) x rank = 30 x 55
        # V index = j*p + k
        # j=0, k=2b -> V[0*10 + 2b] = V[2b]
        # j=0, k=2b+1 -> V[2b+1]
        # j=1, k=2b -> V[10+2b]
        # j=1, k=2b+1 -> V[10+2b+1]
        # j=2, k=2b -> V[20+2b]
        # j=2, k=2b+1 -> V[20+2b+1]
        
        v_indices = [2*b, 2*b+1, 10+2*b, 10+2*b+1, 20+2*b, 20+2*b+1]
        for i in range(6):
            for k in range(11):
                V[v_indices[i]][offset + k] = block_V[i][k]
                
        # W indices for block b
        # C rows 0,1; cols 2b, 2b+1
        # W index = k*n + i
        # i=0, k=2b -> W[2b*2 + 0] = W[4b]
        # i=0, k=2b+1 -> W[2b*2 + 1] = W[4b+1]
        # i=1, k=2b -> W[(2b+1)*2 + 0] = W[4b+2]
        # i=1, k=2b+1 -> W[(2b+1)*2 + 1] = W[4b+3]
        
        w_indices = [4*b, 4*b+1, 4*b+2, 4*b+3]
        for i in range(4):
            for k in range(11):
                W[w_indices[i]][offset + k] = block_W[i][k]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
