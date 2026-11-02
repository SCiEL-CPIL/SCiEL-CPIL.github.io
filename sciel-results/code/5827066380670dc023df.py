"""Construct a rank-12 tensor decomposition for 3x3 * 3x4 matrix multiplication."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 4
    rank = 12
    
    U = [[0] * rank for _ in range(n * m)]   # 9 rows
    V = [[0] * rank for _ in range(m * p)]   # 12 rows
    W = [[0] * rank for _ in range(p * n)]   # 12 rows
    
    # Terms 0-8: schoolbook for (i,j,k) with i,j,k in {0,1,2}
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):  # k=0,1,2 only
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Terms 9-11: for the k=3 column, one term per j value
    # Term for j0: U[i, j0, r]=1 for all i, V[j0, 3, r]=1, W[3, i, r]=1 for all i
    for j0 in range(m):
        r = col
        for i in range(n):
            U[i * m + j0][r] = 1
            W[3 * n + i][r] = 1
        V[j0 * p + 3][r] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
