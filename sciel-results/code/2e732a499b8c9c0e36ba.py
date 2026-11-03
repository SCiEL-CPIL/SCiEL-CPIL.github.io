"""Matrix multiplication 3,3,8 decomposition."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition
    n, m, p = DIMENSIONS
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
