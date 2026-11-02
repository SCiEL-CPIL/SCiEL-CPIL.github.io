"""Matrix multiplication tensor decomposition for 3x3 times 3x5 matrices."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total = n * m * p
    rows_u = n * m
    rows_v = m * p
    rows_w = p * n

    U = [[0.0] * total for _ in range(rows_u)]
    V = [[0.0] * total for _ in range(rows_v)]
    W = [[0.0] * total for _ in range(rows_w)]

    # Precompute all (i, j, k) triples and their corresponding indices
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i, j, k))

    for idx, (i, j, k) in enumerate(triples):
        row_u = i * m + j
        row_v = j * p + k
        row_w = k * n + i
        U[row_u][idx] = 1.0
        V[row_v][idx] = 1.0
        W[row_w][idx] = 1.0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
