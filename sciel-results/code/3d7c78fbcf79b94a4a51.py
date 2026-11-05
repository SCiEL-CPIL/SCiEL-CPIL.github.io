"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    # Kronecker product structure: (2x1x2) ⊗ (1x3x5)
    # Rank = rank(2x1x2) * rank(1x3x5) = 2 * 15 = 30
    rank = 30
    U = [[0] * rank for _ in range(n * m)]   # 6 x 30
    V = [[0] * rank for _ in range(m * p)]   # 30 x 30
    W = [[0] * rank for _ in range(p * n)]   # 20 x 30

    col = 0
    # T1 (2x1x2) rank-2 terms: for i in {0,1}
    # T2 (1x3x5) rank-15 terms: for j in {0,1,2}, k in {0..4}
    # Kronecker term (i, j, k):
    #   U[3*i + j] = 1
    #   V[10*j + 5*i + k] = 1
    #   W[2*(5*i + k) + i] = 1
    for i in range(2):
        for j in range(3):
            for k in range(5):
                u_idx = 3 * i + j
                v_idx = 10 * j + 5 * i + k
                w_idx = 2 * (5 * i + k) + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
