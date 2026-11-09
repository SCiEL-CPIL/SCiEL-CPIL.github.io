"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction

import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
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


def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _tensor_from_decomp(U, V, W):
    """Compute tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] using einsum."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    return np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)


def _verify_decomposition(decomp, T, tol=1e-9):
    """Check if the decomposition exactly reproduces T."""
    U, V, W = decomp['U'], decomp['V'], decomp['W']
    T_approx = _tensor_from_decomp(U, V, W)
    return np.allclose(T_approx, T, atol=tol, rtol=0)


def _try_find_smaller_decomposition(n, m, p, deadline):
    """
    Placeholder for future search. 
    Currently, it returns the schoolbook decomposition.
    A structural search for rank < n*m*p is computationally expensive 
    and unlikely to succeed for 3x4x8 within the time limit without 
    specific algebraic insights.
    """
    # For now, we just return the schoolbook rank and decomp
    # to ensure we have a valid solution.
    return n * m * p, _schoolbook(n, m, p)


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave 40s buffer for the 150s limit
    
    # Try to find a smaller decomposition
    rank, decomp = _try_find_smaller_decomposition(n, m, p, deadline)
    
    # Verify the decomposition to ensure it is correct
    T = _build_target_tensor(n, m, p)
    if not _verify_decomposition(decomp, T):
        # Fallback to schoolbook if verification fails (should not happen)
        decomp = _schoolbook(n, m, p)
    
    return decomp
# EVOLVE-BLOCK-END
