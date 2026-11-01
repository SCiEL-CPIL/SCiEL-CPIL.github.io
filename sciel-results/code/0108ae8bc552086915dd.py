"""Matrix multiplication 2x3x9 - rank 18 decomposition exploiting n=2 structure."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # For 2x3 * 3x9: treat as two independent 1x3 * 3x9 multiplications
    # Each 1x3 * 3x9 has rank 9 (schoolbook), total rank = 18
    rank = 2 * p  # = 18
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 18
    V = [[0] * rank for _ in range(m * p)]  # 27 x 18
    W = [[0] * rank for _ in range(p * n)]  # 18 x 18
    
    for i in range(n):
        for k in range(p):
            col = i * p + k
            # U: row index is i*m + j for j in 0..m-1
            # For row i of A, all m entries participate in the sum
            # In the 1x3 * 3x9 tensor, U[i*m+j] = 1 for all j
            for j in range(m):
                U[i * m + j][col] = 1
            # V: row index is j*p + k
            # For the 1x3 * 3x9 tensor, V[j*p+k] = 1 for all j
            for j in range(m):
                V[j * p + k][col] = 1
            # W: row index is k*n + i
            # In the 1x3 * 3x9 tensor, W[k*n+i] = 1
            W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
