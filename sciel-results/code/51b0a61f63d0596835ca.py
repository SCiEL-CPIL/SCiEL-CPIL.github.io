"""Matrix multiplication 2x3x6 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Rank 9 decomposition by splitting the 6 columns of B into three 2-column blocks.
    # For each block t in {0,1,2}, we use the rank-3 decomposition of 2x3x2 matrix mult.
    n, m, p = 2, 3, 6
    rank = 9
    U = [[0] * rank for _ in range(n * m)]  # 6 x 9
    V = [[0] * rank for _ in range(m * p)]  # 18 x 9
    W = [[0] * rank for _ in range(p * n)]  # 12 x 9

    for t in range(3):
        for j in range(m):  # j in {0,1,2}
            c = 3 * t + j
            for i in range(n):  # i in {0,1}
                U[i * m + j][c] = 1
            for k_prime in range(2):  # k' in {0,1}
                k_global = 2 * t + k_prime
                V[j * p + k_global][c] = 1
                W[k_global * n + i][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
