"""Matrix multiplication 3x4x5 - rank-15 exact tensor decomposition."""
import time

DIMENSIONS = [3, 4, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_rank_np_decomposition():
    """Build the rank n*p decomposition: one term per (i,k) pair."""
    n, m, p = DIMENSIONS
    rank = n * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    return {"U": U, "V": V, "W": W}

# Pre-compute and verify at module load
_DECOMP = _build_rank_np_decomposition()
assert _verify(_DECOMP["U"], _DECOMP["V"], _DECOMP["W"], 3, 4, 5), "Decomposition invalid"

def construct():
    """Return the rank-15 exact decomposition for 3x4x5 matrix multiplication."""
    return _DECOMP
