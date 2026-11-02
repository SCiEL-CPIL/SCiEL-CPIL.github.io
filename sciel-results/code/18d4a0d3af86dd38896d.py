"""Rank-12 decomposition for multiplying a 2x3 matrix by a 3x10 matrix."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    # Decompose C (n x p) = A (n x m) * B (m x p), n=2, m=3, p=10.
    # The 10 output columns split into two blocks of 5 (k0 in [0,5), [5,10)).
    # For a fixed row i, block b, C[i, b*5+s] = sum_j A[i,j] * B[j, b*5+s].
    # The M_{1,3,5} sub-tensor has rank 3: term j uses u=e_j, v=row j of B,
    # w = e_{b*5+s} (a single output column). Repeating for the 2 rows gives
    # rank 3 (i=0) + 3 (i=1) + 3 (i=0) + 3 (i=1) = 12.
    n, m, p = DIMENSIONS
    half = 5
    rank = 2 * n * m  # = 12

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for b in range(2):        # output block: columns b*5 .. b*5+4
        for i in range(n):    # row of the result
            for j in range(m):# index of A[i,j] and B[j, *]
                U[i * m + j][col] = 1
                for s in range(half):
                    V[j * p + (b * half + s)][col] = 1
                W[(b * half + s) * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
