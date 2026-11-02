"""Matrix multiplication 4,4,5 - rank 68 decomposition."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-68 decomposition for 4x4 * 4x5 -> 4x5.
    
    Split B into B[:, :4] (4x4) and B[:, 4:5] (4x1).
    - 4x4 * 4x4 -> 4x4: schoolbook rank 64
    - 4x4 * 4x1 -> 4x1: schoolbook rank 4
    Total rank: 68
    
    Index conventions:
    - U has shape (n*m, rank) = (16, rank): U[i*m+j, col]
    - V has shape (m*p, rank) = (20, rank): V[j*p+k, col]
    - W has shape (p*n, rank) = (20, rank): W[k*n+i, col]
    
    For each bilinear form c_{col}: sum_{i,j,k} W[k*n+i, col] * U[i*m+j, col] * V[j*p+k, col]
    should equal C[i,k] = sum_j A[i,j] * B[j,k]
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    rank = 68
    
    U = [[0] * rank for _ in range(n * m)]  # 16 rows
    V = [[0] * rank for _ in range(m * p)]  # 20 rows
    W = [[0] * rank for _ in range(p * n)]  # 20 rows
    
    col = 0
    # Part 1: 4x4 * 4x4 (k = 0..3), schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(4):  # only first 4 columns of B
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Part 2: 4x4 * 4x1 (k = 4), schoolbook
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
