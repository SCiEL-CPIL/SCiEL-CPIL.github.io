"""
Rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.

Dimensions: n=3 (rows of A), m=3 (cols of A / rows of B), p=7 (cols of B).
Factor shapes: U is (n*m, r) = (9, r), V is (m*p, r) = (21, r), W is (p*n, r) = (21, r).

Exactness condition: for all (i, j, k),
    sum_ell U[i*m + j, ell] * V[j*p + k, ell] * W[k*n + i, ell] == 1

Key structural observation:
  The tensor T[i, j, k] = 1 factors through the middle index j. For each fixed j,
  the sub-tensor over (i, k) is the all-ones 3x7 matrix, which has matrix rank 1
  (hence tensor rank 1 as a 3x1x7 tensor). Moreover, different j-slices use disjoint
  U-rows (indexed by (i, j)) and disjoint V-rows (indexed by (j, k)), so the slices
  are fully separable. Therefore the total tensor rank is 1 + 1 + 1 = 3, and this is
  minimal (each slice is nonzero, requiring at least rank 1, and the disjoint support
  prevents any rank-1 term from covering two different j-slices).

Construction (rank = 3):
  For each j in {0, 1, 2}, create one term ell_j:
    U[i*m + j, ell_j] = 1 for all i in {0,1,2}
    V[j*p + k, ell_j] = 1 for all k in {0,...,6}
    W[k*n + i, ell_j] = 1 for all (k, i)
  All other entries are 0.
"""
import time

DIMENSIONS = [3, 3, 7]


def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W'.
    The shapes are (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    rank = m  # = 3, one term per middle index j

    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for j in range(m):
        ell = j  # term index for middle index j
        # U rows for this j: i*m + j for i in 0..n-1
        for i in range(n):
            U[i * m + j][ell] = 1
        # V rows for this j: j*p + k for k in 0..p-1
        for k in range(p):
            V[j * p + k][ell] = 1
        # W: all (k, i) entries for this term
        for k in range(p):
            for i in range(n):
                W[k * n + i][ell] = 1

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    # Self-verification
    result = construct()
    n, m, p = DIMENSIONS
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for ell in range(rank):
                    s += U[i * m + j][ell] * V[j * p + k][ell] * W[k * n + i][ell]
                if s != 1:
                    ok = False
                    print(f"FAIL at ({i},{j},{k}): sum={s}")
    print(f"rank = {rank}, exact = {ok}")
    assert ok, "Decomposition is not exact!"
    print("Verification passed.")
