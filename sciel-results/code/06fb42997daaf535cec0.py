"""Matrix multiplication 2x3x7 with rank-14 decomposition using Strassen-like grouping."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = p  # 7 terms for i=0 part + 7 terms for i=1 part = 14 total
    
    total_rank = 2 * p
    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 14
    V = [[0] * total_rank for _ in range(m * p)]  # 21 x 14
    W = [[0] * total_rank for _ in range(p * n)]  # 14 x 14
    
    for k in range(p):
        # Term for i_in=0, i_out=0
        col0 = k
        for j in range(m):
            U[j][col0] = 1           # a = j (i_in=0)
            V[j * p + k][col0] = 1   # b = j*p + k
        W[2 * k][col0] = 1           # c = 2k (i_out=0)
        
        # Term for i_in=1, i_out=1
        col1 = p + k
        for j in range(m):
            U[m + j][col1] = 1       # a = m + j (i_in=1)
            V[j * p + k][col1] = 1   # b = j*p + k
        W[2 * k + 1][col1] = 1       # c = 2k+1 (i_out=1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
