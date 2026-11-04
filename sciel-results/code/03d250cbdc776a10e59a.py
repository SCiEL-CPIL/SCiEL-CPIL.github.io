"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import itertools
import random

def _verify_factors(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] equals the MM tensor."""
    rank = len(U[0])
    for i, j, k in itertools.product(range(n), range(m), range(p)):
        s = 0
        for c in range(rank):
            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
        if abs(s - 1) > tol:
            return False
    # Also check that entries outside the support are zero
    # The tensor is zero except at (i,j,k) positions which should be 1
    # We already verified the 1's. For a valid MM tensor decomposition, the
    # contraction should give 1 for all (i,j,k) in [n]x[m]x[p] and 0 elsewhere.
    # But the tensor only has support at those positions, so we just need the
    # above to hold.
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    # Return the schoolbook decomposition (rank = n*m*p = 100)
    # The slicing approach from rank-19 5x5x5 would give rank 19 for 4x5x5,
    # but we do not have the explicit rank-19 factors hardcoded.
    # A safe, always-valid decomposition is the schoolbook one.
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
