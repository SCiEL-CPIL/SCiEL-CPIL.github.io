"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    # Rank-18 decomposition: 3 independent 2x3x3 blocks
    rank = 18
    U = [[0] * rank for _ in range(n * m)]   # 6 x 18
    V = [[0] * rank for _ in range(m * p)]   # 27 x 18
    W = [[0] * rank for _ in range(p * n)]   # 18 x 18
    
    for g in range(3):
        for i_prime in range(n):
            for c in range(m):
                col = g * (n * m) + i_prime * m + c
                for j in range(m):
                    if j == c:
                        U[i_prime * m + j][col] = 1
                for j in range(m):
                    for k in range(3):
                        if k == c:
                            V[j * p + (3 * g + k)][col] = 1
                for k in range(3):
                    for i in range(n):
                        if k == c and i == i_prime:
                            W[(3 * g + k) * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
