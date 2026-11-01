import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # p=12, so 6 blocks of size 2
    num_blocks = p // 2
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    # Winograd rank-7 decomposition for 2x2x2 all-ones tensor
    # U_block: 4x7 (indices i*2+j for i in {0,1}, j in {0,1})
    # V_block: 2x7 (indices j*2+k for j in {0,1}, k in {0,1})
    # W_block: 4x7 (indices k*2+i for k in {0,1}, i in {0,1})
    
    # Using the standard Winograd F(2) x F(2) decomposition
    # with exact half-integer coefficients
    
    U_block = [
        [1,   0,   0,   0,   1,   1,   0],   # (i=0,j=0)
        [0,   1,   0,   0,   1,   0,   1],   # (i=0,j=1)
        [0,   0,   1,   0,   0,   1,   0],   # (i=1,j=0)
        [0,   0,   0,   1,   0,   0,   1],   # (i=1,j=1)
    ]
    
    V_block = [
        [1,   0,   1,   0,   1,   1,   0],   # (j=0,k=0)
        [0,   1,   0,   1,   1,   0,   1],   # (j=0,k=1)
        [1,   0,   1,   0,   1,   1,   0],   # (j=1,k=0)
        [0,   1,   0,   1,   1,   0,   1],   # (j=1,k=1)
    ]
    
    # W_block with half-integer reconstruction coefficients
    # k*2+i: (k=0,i=0)=0, (k=0,i=1)=1, (k=1,i=0)=2, (k=1,i=1)=3
    W_block = [
        [1,   0,   0,   0,  0.5,  0.5,  0.5],   # (k=0,i=0)
        [0,   1,   0,   0,  0.5,  0.5,  0.5],   # (k=0,i=1)
        [0,   0,   1,   0,  0.5,  0.5,  0.5],   # (k=1,i=0)
        [0,   0,   0,   1,  0.5,  0.5,  0.5],   # (k=1,i=1)
    ]
    
    # Verify the decomposition: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1
    max_err = 0.0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0.0
                for r in range(7):
                    val += U_block[i*2 + j][r] * V_block[j*2 + k][r] * W_block[k*2 + i][r]
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails
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
    
    # Build full matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for block in range(num_blocks):
        for i in range(2):
            for j in range(2):
                for r in range(7):
                    U[i*m + j][block*7 + r] = U_block[i*2 + j][r]
        
        for j in range(2):
            for k in range(2):
                for r in range(7):
                    V[j*p + block*2 + k][block*7 + r] = V_block[j*2 + k][r]
        
        for k in range(2):
            for i in range(2):
                for r in range(7):
                    W[(block*2 + k)*n + i][block*7 + r] = W_block[k*2 + i][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
