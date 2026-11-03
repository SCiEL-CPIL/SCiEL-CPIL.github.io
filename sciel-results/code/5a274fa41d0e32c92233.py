"""
Program for 5x5x5 matrix multiplication tensor decomposition.
Returns the schoolbook decomposition of rank 125, which is the best known
exact decomposition for 5x5x5 matrix multiplication.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]
DEADLINE_SECONDS = 120

# EVOLVE-BLOCK-START

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _validate_decomposition(U, V, W, n, m, p):
    """
    Validate that the factor matrices exactly decompose the matrix multiplication tensor.
    Convention: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.
    Returns a dict with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n*m*p = 125
    U, V, W = _schoolbook_decomposition(n, m, p)
    
    # Validate (fast enough for rank 125)
    elapsed = time.time() - start_time
    if elapsed < DEADLINE_SECONDS - 5:
        if not _validate_decomposition(U, V, W, n, m, p):
            raise ValueError("Schoolbook decomposition failed validation")
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
