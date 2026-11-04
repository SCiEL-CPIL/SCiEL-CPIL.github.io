"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    The schoolbook approach uses rank n*m*p = 4*5*6 = 120.
    
    We try to use known low-rank decompositions for sub-blocks.
    
    For 4x4x4, the best known is 49 (Beyls).
    For 4x5x5, we can decompose as 4x4x4 + 4x1x5, etc.
    
    However, without a verified low-rank decomposition in hand, the safest
    is to return the schoolbook decomposition of rank 120.
    
    Let's try a block decomposition approach:
    Split 4x5 * 5x6 into blocks.
    
    Actually, let's try to use Strassen-like ideas for 4x4 blocks.
    
    A simpler approach: just return the schoolbook. The rank is 120.
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    rank = n * m * p  # 120
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
