"""Matrix multiplication 2x2x16 with Strassen-based rank-56 decomposition."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct rank-56 decomposition for 2x2 * 2x16 matrix multiplication.
    
    The 2x16 matrix has 8 independent column pairs. Each 2x2x2 multiplication
    uses Strassen's rank-7 algorithm, giving total rank 8*7 = 56.
    
    U: shape (n*m, rank) = (4, 56)
    V: shape (m*p, rank) = (32, 56)  
    W: shape (p*n, rank) = (32, 56)
    
    Indexing:
    - U[i*m+j, c] where i in [0,2), j in [0,2)
    - V[j*p+k, c] where j in [0,2), k in [0,16)
    - W[k*n+i, c] where k in [0,16), i in [0,2)
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 8 blocks of 2 columns each
    block_rank = 7
    rank = num_blocks * block_rank  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    # Strassen's 7 products for 2x2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # For each block of 2 columns (k, k+1):
    # A indices: i in {0,1}, j in {0,1} -> U index: i*m + j
    # B indices: j in {0,1}, k in {0,1} -> V index: j*p + (block*2 + k)
    # C indices: k in {0,1}, i in {0,1} -> W index: (block*2 + k)*n + i
    
    for block in range(num_blocks):
        base_col = block * 2
        base_rank = block * block_rank
        
        # M1: (a11+a22)(b11+b22)
        # U: a11=1, a22=1
        U[0][base_rank + 0] = 1  # a11 (i=0,j=0)
        U[3][base_rank + 0] = 1  # a22 (i=1,j=1)
        # V: b11=1, b22=1
        V[0 * p + base_col][base_rank + 0] = 1      # b11 (j=0,k=0)
        V[1 * p + base_col + 1][base_rank + 0] = 1  # b22 (j=1,k=1)
        # W: c11 += 1, c22 += 1
        W[base_col * n + 0][base_rank + 0] = 1    # c11 (k=0,i=0)
        W[(base_col + 1) * n + 1][base_rank + 0] = 1  # c22 (k=1,i=1)
        
        # M2: (a21+a22)b11
        # U: a21=1, a22=1
        U[1][base_rank + 1] = 1  # a21 (i=0,j=1)
        U[3][base_rank + 1] = 1  # a22 (i=1,j=1)
        # V: b11=1
        V[0 * p + base_col][base_rank + 1] = 1  # b11 (j=0,k=0)
        # W: c21 += 1
        W[base_col * n + 1][base_rank + 1] = 1  # c21 (k=0,i=1)
        
        # M3: a11(b12-b22)
        # U: a11=1
        U[0][base_rank + 2] = 1  # a11 (i=0,j=0)
        # V: b12=1, b22=-1
        V[0 * p + base_col + 1][base_rank + 2] = 1   # b12 (j=0,k=1)
        V[1 * p + base_col + 1][base_rank + 2] = -1  # b22 (j=1,k=1)
        # W: c12 += 1
        W[(base_col + 1) * n + 0][base_rank + 2] = 1  # c12 (k=1,i=0)
        
        # M4: a22(b21-b11)
        # U: a22=1
        U[3][base_rank + 3] = 1  # a22 (i=1,j=1)
        # V: b21=1, b11=-1
        V[1 * p + base_col][base_rank + 3] = 1   # b21 (j=1,k=0)
        V[0 * p + base_col][base_rank + 3] = -1  # b11 (j=0,k=0)
        # W: c11 += 1, c21 += 1
        W[base_col * n + 0][base_rank + 3] = 1    # c11 (k=0,i=0)
        W[base_col * n + 1][base_rank + 3] = 1    # c21 (k=0,i=1)
        
        # M5: (a11+a12)b22
        # U: a11=1, a12=1
        U[0][base_rank + 4] = 1  # a11 (i=0,j=0)
        U[2][base_rank + 4] = 1  # a12 (i=1,j=0)
        # V: b22=1
        V[1 * p + base_col + 1][base_rank + 4] = 1  # b22 (j=1,k=1)
        # W: c11 -= 1, c12 += 1
        W[base_col * n + 0][base_rank + 4] = -1     # c11 (k=0,i=0)
        W[(base_col + 1) * n + 0][base_rank + 4] = 1  # c12 (k=1,i=0)
        
        # M6: (a21-a11)(b11+b12)
        # U: a21=1, a11=-1
        U[1][base_rank + 5] = 1   # a21 (i=0,j=1)
        U[0][base_rank + 5] = -1  # a11 (i=0,j=0)
        # V: b11=1, b12=1
        V[0 * p + base_col][base_rank + 5] = 1      # b11 (j=0,k=0)
        V[0 * p + base_col + 1][base_rank + 5] = 1  # b12 (j=0,k=1)
        # W: c22 += 1
        W[(base_col + 1) * n + 1][base_rank + 5] = 1  # c22 (k=1,i=1)
        
        # M7: (a12-a22)(b21+b22)
        # U: a12=1, a22=-1
        U[2][base_rank + 6] = 1   # a12 (i=1,j=0)
        U[3][base_rank + 6] = -1  # a22 (i=1,j=1)
        # V: b21=1, b22=1
        V[1 * p + base_col][base_rank + 6] = 1      # b21 (j=1,k=0)
        V[1 * p + base_col + 1][base_rank + 6] = 1  # b22 (j=1,k=1)
        # W: c11 += 1
        W[base_col * n + 0][base_rank + 6] = 1  # c11 (k=0,i=0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
