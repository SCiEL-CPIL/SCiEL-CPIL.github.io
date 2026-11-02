"""Matrix multiplication 4x4 x 4x9 with schoolbook decomposition."""
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    rank = n * m * p  # 144
    U = [[0] * rank for _ in range(n * m)]  # 16 x 144
    V = [[0] * rank for _ in range(m * p)]  # 36 x 144
    W = [[0] * rank for _ in range(p * n)]  # 36 x 144

    col = 0

    def a_idx(i, j):
        return i * m + j

    def b_idx(j, k):
        return j * p + k

    def c_idx(k, i):
        return k * n + i

    # Schoolbook decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[a_idx(i, j)][col] = 1
                V[b_idx(j, k)][col] = 1
                W[c_idx(k, i)][col] = 1
                col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
