import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the n×m×p matrix multiplication tensor."""
    rank = U.shape[1]
    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    # Compute decomposition tensor
    T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T_dec, T, atol=tol)

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

def _find_structured_rank1(T, n, m, p):
    """
    Find a structured rank-1 term that matches a subset of the tensor entries.
    Returns (scale, u, v, w) or None.
    """
    # Try different support patterns
    # A rank-1 term has u[i*m+j] * v[j*p+k] * w[k*n+i] = T[i*m+j, j*p+k, k*n+i]
    # for (i,j,k) in the support.
    
    # For simplicity, try patterns where the support is a "rectangle"
    # in (i,j,k) space: i in I, j in J, k in K
    # Then u[i*m+j] = a[i]*b[j], v[j*p+k] = b[j]*c[k], w[k*n+i] = c[k]*a[i]
    # and the product is a[i]*b[j]^2*c[k]*a[i] = a[i]^2*b[j]^2*c[k]
    # This should equal T[i*m+j, j*p+k, k*n+i] = 1 for (i,j,k) in I×J×K
    
    # So we need a[i]^2 * b[j]^2 * c[k] = 1 for all (i,j,k) in I×J×K
    # This implies a[i] = constant, b[j] = constant, c[k] = constant
    # So the only rectangular patterns that work are with constant values.
    
    # This means the only pure rank-1 terms are the schoolbook ones.
    # To get Strassen-like improvements, we need more complex patterns.
    
    return None

def _try_structured_2x3x6(n, m, p):
    """
    Try to construct a lower-rank decomposition for 2x3x6.
    """
    # The peeling approach with pure rank-1 terms won't work better than schoolbook.
    # We need to find non-pure rank-1 terms.
    # For now, return None.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # First, try to find a structured lower-rank decomposition
    best_solution = None
    best_rank = n * m * p  # 36
    
    # Try structured approach
    if time.time() < deadline - 30:
        result = _try_structured_2x3x6(n, m, p)
        if result is not None:
            U, V, W, R = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_solution = (U, V, W, R)
                best_rank = R
    
    # If no improvement found, fall back to schoolbook
    if best_solution is None:
        U, V, W, rank = _schoolbook(n, m, p)
        best_solution = (U, V, W, rank)
        best_rank = rank
    
    U, V, W, R = best_solution
    
    # Convert to lists for return
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

