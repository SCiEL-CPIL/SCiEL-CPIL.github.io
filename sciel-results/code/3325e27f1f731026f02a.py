"""Trivial rank-32 decomposition for 2x4 by 4x4 matrix multiplication."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """Trivial rank-32 decomposition for 2x4 * 4x4 matrix multiplication.
    
    Uses the standard identity: C[i,j] = sum_k A[i,k] * B[k,j]
    Each term A[i,k]*B[k,j] is one rank-1 component.
    Total rank = n*m*p = 2*4*4 = 32.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = n * m * p  # 32
    U = [[0] * rank for _ in range(n * m)]  # 8 x 32
    V = [[0] * rank for _ in range(m * p)]  # 16 x 32
    W = [[0] * rank for _ in range(p * n)]  # 8 x 32

    for i in range(n):
        for k in range(m):
            for j in range(p):
                s = i * (m * p) + k * p + j
                U[i * m + k][s] = 1
                V[k * p + j][s] = 1
                W[j * n + i][s] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
