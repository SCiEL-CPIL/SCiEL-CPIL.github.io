"""Matrix multiplication 3x4x7 via block partitioning."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    import time
    t0 = time.time()
    DEADLINE = 130.0

    n, m, p = DIMENSIONS
    n1, n2 = 1, 2
    m1, m2 = 3, 4

    r1 = n1 * m1 * m2
    r2 = n2 * m1 * m2
    rank = r1 + r2

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Block 1: rows 0 of A (1x4) times cols 0..m1-1 of B (4xm1)
    # A_top: row i in [0, n1), col j in [0, m1)
    # B_left: row j in [0, m1), col k in [0, m2)
    # Output: row i in [0, n1), col k in [0, m2)
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(m2):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    if time.time() > DEADLINE - 5:
        return {"U": U, "V": V, "W": W}

    # Block 2: rows n1..n-1 of A (n2x4) times cols m1..p-1 of B (4xm2)
    # A_bot: row i in [n1, n), col j in [0, m1)
    # B_right: row j in [0, m1), col k in [m1, p)
    # Output: row i in [n1, n), col k in [m1, p)
    for i in range(n1, n):
        for j in range(m1):
            for k in range(m1, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
