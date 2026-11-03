import time
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create a schoolbook decomposition for M_{n, m, p}
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # For M_{3,5,7}, try to find a decomposition with rank < 105
    # One approach: use the fact that we can write the multiplication as
    # a sum of rank-1 terms more efficiently
    
    # Let's try a structured approach:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We can group the j indices and try to share computations
    
    # For now, let's return schoolbook as baseline, but try to improve
    U, V, W = schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
