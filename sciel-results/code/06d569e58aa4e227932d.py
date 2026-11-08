"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 120.0
    n, m, p = 2, 3, 3
    rank = 17

    # M = [[a1, a2, a3], [a4, a5, a6]] (A)  rows i in 0..1, cols j in 0..2
    # B = rows j in 0..2, cols k in 0..2 ;  b(j*3+k)
    #
    # Each term is (alpha_row_i, beta_row_j) with a coefficient.
    # alpha_row_i: 3-tuple giving coeffs on [a1,a2,a3] (A row i)
    # beta_row_j : 3-tuple giving coeffs on [b_j0, b_j1, b_j2] (B row j)
    terms = [
        ((1,1,0), (1,1,0)),
        ((1,0,0), (1,0,0)),
        ((1,0,0), (0,1,1)),
        ((0,1,0), (1,0,0)),
        ((0,1,0), (0,1,1)),
        ((0,1,1), (1,0,0)),
        ((1,1,1), (0,1,0)),
        ((1,1,0), (0,1,0)),
        ((0,0,1), (0,1,0)),
        ((0,0,1), (1,0,0)),
        ((0,0,1), (0,1,1)),
        ((1,1,0), (1,1,0)),
        ((0,0,1), (1,0,0)),
        ((0,0,1), (0,1,1)),
        ((1,1,0), (1,1,0)),
        ((0,1,0), (1,0,0)),
        ((0,1,0), (0,1,1)),
    ]

    U = [[0] * rank for _ in range(n * m)]   # index i*m + j
    V = [[0] * rank for _ in range(m * p)]   # index j*p + k
    W = [[0] * rank for _ in range(p * n)]   # index k*n + i (trace convention)

    for c, (arow, brow) in enumerate(terms):
        for i in range(n):
            if arow[i]:
                U[i * m + 0][c] = arow[i]  # placeholder, replaced below
        # rebuild properly: U row (i, j) for each nonzero coefficient on A[i][j]
    # Redo cleanly
    for c in range(rank):
        for i in range(n):
            arow = terms[c][0]
            for j in range(m):
                if arow[j]:
                    U[i * m + j][c] = arow[j]
        brow = terms[c][1]
        for j in range(m):
            for k in range(p):
                if brow[k]:
                    V[j * p + k][c] = brow[k]
        # W: coefficient for C[i][k] = sum_c U[i,m+j] V[j,p+k]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += U[i * m + j][c] * V[j * p + k][c]
                W[k * n + i][c] = s

    # Verify tensor identity: for every (i,j,k), C[i][k] = sum_c U[i*m+j][c]*V[j*p+k][c]
    # must equal delta(i,k) (the "1" in the multiplication tensor at position (i,j,k)).
    def _check():
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c]
                    expect = 1 if i == k else 0
                    if s != expect:
                        return False
        return True

    if time.time() < DEADLINE - 2.0:
        if not _check():
            # Fallback to schoolbook rank 18 (always valid) to avoid an invalid score.
            n, m, p = 2, 3, 3
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
            return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
