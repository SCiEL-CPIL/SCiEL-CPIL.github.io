"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Implementing the schoolbook decomposition for 3x3x5 matrix multiplication.
    
    The directive suggested a rank-14 decomposition, which is mathematically impossible
    (lower bound for 3x3x3 is 19). The correct direct-sum decomposition into 3x3x3 and 3x3x2
    blocks yields a rank of 23 + 18 = 41, but requires a specific rank-23 decomposition
    for the 3x3x3 block which is not trivially available.
    
    The current implementation uses the standard schoolbook algorithm with rank 45,
    which is guaranteed to be correct and valid.
    """
    n, m, p = DIMENSIONS  # 3, 3, 5
    rank = n * m * p  # 45
    U = [[0] * rank for _ in range(n * m)]  # 9 x 45
    V = [[0] * rank for _ in range(m * p)]  # 15 x 45
    W = [[0] * rank for _ in range(p * n)]  # 15 x 45
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
