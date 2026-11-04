"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    blocks = [
        (0, 0),
        (0, 2),
        (2, 0),
        (2, 2)
    ]
    
    col = 0
    for j_off, k_off in blocks:
        for t in range(7):
            if t == 0:
                U[0 * m + j_off][col] += 1
                U[1 * m + (j_off + 1)][col] += 1
                V[j_off * p + k_off][col] += 1
                V[(j_off + 1) * p + (k_off + 1)][col] += 1
                W[k_off * n + 0][col] += 1
                W[(k_off + 1) * n + 1][col] += 1
            elif t == 1:
                U[1 * m + j_off][col] += 1
                U[1 * m + (j_off + 1)][col] += 1
                V[j_off * p + k_off][col] += 1
                W[(k_off + 1) * n + 0][col] += 1
            elif t == 2:
                U[0 * m + j_off][col] += 1
                V[j_off * p + (k_off + 1)][col] += 1
                V[(j_off + 1) * p + (k_off + 1)][col] -= 1
                W[k_off * n + 1][col] += 1
            elif t == 3:
                U[1 * m + (j_off + 1)][col] += 1
                V[(j_off + 1) * p + k_off][col] += 1
                V[j_off * p + k_off][col] -= 1
                W[k_off * n + 0][col] -= 1
                W[(k_off + 1) * n + 0][col] += 1
            elif t == 4:
                U[0 * m + j_off][col] += 1
                U[0 * m + (j_off + 1)][col] += 1
                V[(j_off + 1) * p + (k_off + 1)][col] += 1
                W[k_off * n + 0][col] += 1
                W[k_off * n + 1][col] += 1
            elif t == 5:
                U[1 * m + j_off][col] += 1
                U[1 * m + (j_off + 1)][col] -= 1
                V[j_off * p + (k_off + 1)][col] += 1
                V[(j_off + 1) * p + (k_off + 1)][col] += 1
                W[(k_off + 1) * n + 1][col] += 1
            elif t == 6:
                U[0 * m + (j_off + 1)][col] += 1
                U[1 * m + j_off][col] -= 1
                V[j_off * p + k_off][col] += 1
                V[(j_off + 1) * p + k_off][col] += 1
                W[k_off * n + 0][col] += 1
                W[(k_off + 1) * n + 1][col] += 1
            col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
