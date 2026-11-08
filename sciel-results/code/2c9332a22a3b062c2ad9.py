"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Try to construct a rank-20 decomposition for 2x3 x 3x9
    # Based on bilinear complexity theory: beta(2,m,p) = 2p + m - 1 for p >= m
    target_rank = 2 * p + m - 1  # 2*9 + 3 - 1 = 20
    
    U = [[0] * target_rank for _ in range(n * m)]  # 6 x 20
    V = [[0] * target_rank for _ in range(m * p)]  # 27 x 20
    W = [[0] * target_rank for _ in range(p * n)]  # 18 x 20
    
    # Construction for 2xm x mxp with rank 2p + m - 1:
    # For k in range(p):
    #   Term 2k: handles (i=0, j=0, k) and related
    #   Term 2k+1: handles (i=1, j=0, k) and related
    # Plus m-1 correction terms
    
    # Let me use a different known construction
    # For 2x3 x 3x9, use the following pattern:
    
    # Actually, let me just use the schoolbook as fallback and try rank 20
    # First, let's verify if rank 20 works with a specific construction
    
    # Known construction for 2xm x mxp:
    # For each k (0..p-1):
    #   c = 2*k: U[0*m+0][c]=1, V[0*p+k][c]=1, W[k*n+0][c]=1
    #   c = 2*k+1: U[1*m+0][c]=1, V[0*p+k][c]=1, W[k*n+1][c]=1
    # This gives 2p terms but only covers j=0
    # We need to handle j=1,2 as well
    
    # Let me think differently. The bilinear complexity of 2xm x mxp is 2p + m - 1.
    # Here's a construction:
    
    # For j in 0..m-1, k in 0..p-1, i in 0..n-1:
    # We need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Construction:
    # For k in 0..p-1:
    #   For i in 0..n-1:
    #     c = i*p + k  (this is 2p terms)
    #     U[i*m+0][c] = 1
    #     V[0*p+k][c] = 1  
    #     W[k*n+i][c] = 1
    #   For j in 1..m-1:
    #     c = 2*p + j - 1  (this is m-1 terms)
    #     U[i*m+j][c] = 1 for all i
    #     V[j*p+k][c] = 1 for all k... no this doesn't work
    
    # I'm not getting the construction right. Let me just use schoolbook rank 54
    # which is guaranteed to work.
    
    rank = n * m * p  # 54
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
# EVOLVE-BLOCK-END
