"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a tensor decomposition for 2x3 by 3x9 matrix multiplication
    using Kronecker product structure.
    
    We view the 3x9 matrix B as three 3x3 blocks. The multiplication
    C = A * B becomes three independent 2x3 by 3x3 multiplications.
    
    For each 2x3 by 3x3 multiplication, we use the schoolbook algorithm
    with rank 18. Total rank is 3 * 18 = 54.
    """
    n, m, p = DIMENSIONS
    
    # Decompose p=9 into p1=3 and p2=3
    p1 = 3
    p2 = 3
    
    # For each block k2 in {0,1,2}, we do a 2x3 by 3x3 multiplication
    # Each has schoolbook rank 18
    rank_per_block = n * m * p1  # 2*3*3 = 18
    total_rank = p2 * rank_per_block  # 3 * 18 = 54
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for k2 in range(p2):
        for i in range(n):
            for j in range(m):
                for k1 in range(p1):
                    k = k1 * p2 + k2  # Map (k1, k2) to k
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
