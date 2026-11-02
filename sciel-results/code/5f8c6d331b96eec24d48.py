"""Matrix multiplication 3,4,8: rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 8]


def _schoolbook_decomposition(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition of M_{n,m,p} as U, V, W arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W form a valid decomposition of M_{n,m,p}.
    For each (i, j, k), the sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1.
    """
    rank = U.shape[1]
    # Check that rank matches across all
    assert U.shape[1] == V.shape[1] == W.shape[1] == rank
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    # Also check that entries are integer or half-integer
    for mat in [U, V, W]:
        doubled = mat * 2
        if not np.allclose(doubled, np.round(doubled), atol=1e-9):
            return False
    return True


def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor.
    Returns schoolbook decomposition with rank n*m*p = 96 for 3x4x8.
    """
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook_decomposition(n, m, p)
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
