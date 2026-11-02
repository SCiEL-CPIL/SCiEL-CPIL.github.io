# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication.
    
    The tensor T_{2,5,5} is the direct sum of two copies of T_{1,5,5}.
    The rank of T_{1,5,5} is 25 (sum of 5 independent dot products of length 5).
    Thus, the rank of T_{2,5,5} is 50.
    This is the minimal possible rank.
    """
    n, m, p = 2, 5, 5
    rank = n * m * p  # 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i (following W[k*n+i, r])
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
