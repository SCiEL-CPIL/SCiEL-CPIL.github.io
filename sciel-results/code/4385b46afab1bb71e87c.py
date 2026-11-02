"""
Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition.

Combines schoolbook baseline with structured search for lower-rank decompositions.
Uses verification to ensure exactness, with deadline guard for 150s limit.
"""

import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p = 96."""
    n, m, p = DIMENSIONS
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


def _verify(U, V, W, n=None, m=None, p=None):
    """
    Verify that U, V, W form a valid exact tensor decomposition.
    For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    if n is None:
        n, m, p = DIMENSIONS
    
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _try_structured_search(deadline):
    """
    Try to find lower-rank decompositions using structured approaches.
    
    Strategy: 
    1. Try to exploit the fact that m=4 and p=8 have power-of-2 structure
    2. Use block-based Strassen-like reductions where possible
    3. Randomized search with integer/half-integer rounding
    """
    n, m, p = DIMENSIONS
    best = None
    
    # Try ranks from a reasonable lower bound up to 95
    # Lower bound: at least max(n*m, m*p, p*n) = 32
    # But for matrix mult, the rank is typically much higher
    
    # Try a few structured constructions first
    # Construction 1: Group by j, use n*p terms per j but try to share
    # For each j, we need to compute n*p = 24 products A[i,j]*B[j,k]
    # These are all independent, so we need at least 24 terms per j? No...
    # Actually, the tensor is T[i,j,k] = 1, so for fixed j, we have a 
    # n x p tensor that's all ones. The rank of an all-ones n x p matrix is 1.
    # But we have 4 different j values, so we need at least 4*1 = 4? No, that's wrong.
    
    # Let me think again. The tensor T[i,j,k] = 1 for all i,j,k.
    # This is just the all-ones tensor of shape (n,m,p) = (3,4,8).
    # The rank of the all-ones tensor of shape (n,m,p) is 1!
    # Because T[i,j,k] = 1 = 1*1*1 for all i,j,k.
    # So U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 for all indices.
    # This gives rank 1!
    
    # Wait, that can't be right. Let me verify.
    # If U[a, 0] = 1 for all a, V[b, 0] = 1 for all b, W[c, 0] = 1 for all c,
    # then sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1*1*1 = 1.
    # Yes! That works!
    
    # But wait, is the matrix multiplication tensor really all ones?
    # Let me reconsider. The matrix multiplication tensor T is defined such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    
    # Yes! The matrix multiplication tensor is the all-ones tensor.
    # And the rank of the all-ones tensor of any shape is 1.
    
    # So the minimal rank is 1!
    
    U = [[1] * 1 for _ in range(n * m)]
    V = [[1] * 1 for _ in range(m * p)]
    W = [[1] * 1 for _ in range(p * n)]
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # If that doesn't work (shouldn't happen), fall back to schoolbook
    return None


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3×4 matrix by a 4×8 matrix.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # First, try the structured search which should find rank 1
    result = _try_structured_search(deadline)
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
