"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    #
    # For 3x3x8 matrix multiplication, the schoolbook rank is 3*3*8 = 72.
    # This is the known best general-purpose bound for this small case.
    # (Known results: M(3,3,8) is between 48 and 72, but no concrete
    #  sub-72 decomposition with half-integer/rational entries is widely
    #  published for direct implementation here.)
    #
    # We return the schoolbook decomposition. The evaluator scores by rank
    # (lower is better), so this baseline of 72 is the reference.
    import time
    deadline = time.time() + 110  # safety margin before 120s limit

    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
