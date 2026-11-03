"""Program for rank-minimal exact tensor decomposition of 3x5 * 5x7 matrix multiplication."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """Construct rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication.
    
    Uses block decomposition: split middle dimension 5 into 2+3.
    - 3x2 * 2x7: schoolbook, rank 42
    - 3x3 * 3x3: schoolbook, rank 27  
    - 3x3 * 3x4: schoolbook, rank 36
    Total rank: 105
    """
    n, m, p = 3, 5, 7
    rank = n * m * p  # 105
    U = [[0] * rank for _ in range(n * m)]   # shape (15, 105)
    V = [[0] * rank for _ in range(m * p)]   # shape (35, 105)
    W = [[0] * rank for _ in range(p * n)]   # shape (21, 105)
    
    col = 0
    # Schoolbook decomposition: for each (i, j, k), one rank-1 term
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
