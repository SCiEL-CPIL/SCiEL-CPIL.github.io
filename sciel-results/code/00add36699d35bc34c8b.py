"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    WFT-based decomposition for 2x4 x 4x7 matrix multiplication.
    
    Strategy: Split the 4-dim middle dimension into two 2-dim blocks.
    For each 2x2 * 2x7 subproblem, use WFT over Z_2 x Z_7.
    
    For 2x2 * 2x7:
    C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    
    We embed this into a convolution on G = Z_2 x Z_7 (order 14).
    The key insight: we can use the character table of Z_2 x Z_7 to diagonalize.
    
    For the Z_2 part: characters are 1, -1
    For the Z_7 part: we need to handle the 7th roots of unity.
    
    However, for exact rational decomposition, we use the following trick:
    Since 7 is prime and we're working with 2x2 blocks, we can use the fact that
    the rank of 2x2*2x7 is at most 2*7 = 14 using a structured decomposition.
    
    Let's use a direct construction:
    For the 2x2*2x7 case, we have:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    Using the sum/difference trick:
    S = A[0,0] + A[0,1], D = A[0,0] - A[0,1]
    S' = A[1,0] + A[1,1], D' = A[1,0] - A[1,1]
    
    This still gives rank 28 for each 2x2*2x7 (no improvement).
    
    Let me try a different approach: use the fact that for n=2, we can achieve
    rank 2*p = 14 for 2x1*1xp type problems, but for 2x2*2x7 it's different.
    
    Actually, let's use a known result: the rank of (2x2)*(2x7) is 14.
    Construction:
    For each k in 0..6, we need to compute:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    We can use 14 terms:
    For k in 0..6:
      Term 2k:   U[0,2k]=1, U[1,2k]=1, V[0,2k]=B[0,k] coeff, V[1,2k]=B[1,k] coeff
      ...
    
    This is getting complex. Let me just use the schoolbook which gives rank 56.
    """
    n, m, p = DIMENSIONS
    
    # Use schoolbook decomposition as baseline
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
# EVOLVE-BLOCK-END
