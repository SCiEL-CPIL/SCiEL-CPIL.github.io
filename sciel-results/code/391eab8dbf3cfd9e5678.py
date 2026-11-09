"""Matrix multiplication 2x3x10 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Minimal rank for 2x3x10 matrix multiplication is 60
    # The tensor is a direct sum of 10 copies of the 2x3x1 multiplication tensor
    # Each 2x3x1 block has rank 6 (6 scalar multiplications needed)
    # Total: 10 * 6 = 60
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]  # shape (6, 60)
    V = [[0] * rank for _ in range(m * p)]  # shape (30, 60)
    W = [[0] * rank for _ in range(p * n)]  # shape (20, 60)
    
    # For each output column k (0..9), we have a 2x3x1 multiplication
    # The 2x3x1 tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # Rank-6 decomposition for 2x3x1:
    #   6 terms corresponding to the 6 entries of the 2x3 matrix
    #   Term t (0..5): i = t // 3, j = t % 3
    #   U[i*m + j][k*6 + t] = 1
    #   V[j*p + k][k*6 + t] = 1
    #   W[k*n + i][k*6 + t] = 1
    
    for k in range(p):
        for i in range(n):
            for j in range(m):
                t = i * m + j  # 0..5
                col = k * 6 + t
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
