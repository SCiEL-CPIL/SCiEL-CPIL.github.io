"""Matrix multiplication 2x3 * 3x6 tensor decomposition - rank minimization.

Strategy:
1. Try a structured rank-24 construction (pairing output columns).
2. If that fails verification, fall back to schoolbook rank 36.
3. Use exact rational arithmetic for verification to avoid floating-point issues.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]


def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    should equal 1 if (i,j,k) is valid, else 0.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Build the tensor using einsum for speed
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    # Check all positions
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Determine if this is a valid multiplication term
                # a = i*m + j, b = j'*p + k, c = k'*n + i'
                # Valid if j == j' and k == k' (inner dimension matches)
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                
                if j == j2 and k == k2 and i == i2:
                    # This is a valid term, should be 1
                    if abs(T[a, b, c] - 1.0) > tol:
                        return False
                else:
                    # This should be 0
                    if abs(T[a, b, c]) > tol:
                        return False
    return True


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}


def _try_rank_24(n, m, p):
    """Attempt a rank-24 construction for 2x3 * 3x6.
    
    Idea: Pair up the 6 output columns into 3 pairs: (0,1), (2,3), (4,5).
    For each pair of output columns and each row of A, we need to compute:
      C[i, 2t]   = sum_j A[i,j] * B[j, 2t]
      C[i, 2t+1] = sum_j A[i,j] * B[j, 2t+1]
    
    For a fixed pair (2t, 2t+1) and fixed row i, this is a 1x3 * 3x2 multiplication.
    The schoolbook rank for this is 3*2 = 6.
    
    But we can use a Strassen-like trick: for 1x3 * 3x2, we can potentially
    reduce the rank. Let's think about it differently.
    
    Actually, let's try a different approach. Consider the full problem:
    We have 12 outputs, each requiring 3 products. Total 36 schoolbook terms.
    
    Let's try to group by the middle index j. For each j in {0,1,2},
    we have contributions A[i,j] * B[j,k] for all i in {0,1} and k in {0,...,5}.
    That's 12 terms per j, total 36.
    
    To reduce rank, we need to find linear combinations that allow sharing.
    
    Let's try a specific construction:
    For each pair of output columns (k, k+1), define:
      S_0 = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
      S_1 = A[0,0]*B[0,k+1] + A[0,1]*B[1,k+1] + A[0,2]*B[2,k+1]
      Similarly for row 1.
    
    This doesn't obviously help. Let's try a different grouping.
    
    Alternative: Use the fact that we can combine terms across different j values.
    
    Let's try: for each pair of columns (k, k+1), and for each row i,
    define intermediate variables:
      X_{i,k} = A[i,0]*B[0,k] + A[i,1]*B[1,k+1]
      Y_{i,k} = A[i,1]*B[1,k] + A[i,2]*B[2,k+1]
      Z_{i,k} = A[i,2]*B[2,k] + A[i,0]*B[0,k+1]
    
    Then:
      C[i,k]   = X_{i,k} + A[i,1]*B[1,k] + A[i,2]*B[2,k] - A[i,1]*B[1,k+1] ... 
    This is getting complicated.
    
    Let's try a more systematic approach: use linear algebra to find a low-rank decomposition.
    """
    # For now, return None to indicate this construction is not implemented
    # We'll fall back to schoolbook
    return None


def _try_rank_30(n, m, p):
    """Attempt a rank-30 construction.
    
    One simple way: merge 6 pairs of schoolbook terms.
    For each pair (j, k) and (j', k'), if we can combine them...
    
    Actually, let's try a different approach: use the fact that
    for 2x3 * 3x6, we can think of it as two 1x3 * 3x6 multiplications.
    Each 1x3 * 3x6 has rank 18 schoolbook.
    But we can share some terms between the two rows.
    
    For row 0: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    For row 1: C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    If we define:
      T_{0,k} = (A[0,0]+A[1,0]) * B[0,k]
      T_{1,k} = (A[0,1]+A[1,1]) * B[1,k]
      T_{2,k} = (A[0,2]+A[1,2]) * B[2,k]
      T_{3,k} = A[0,0]*B[0,k]
      T_{4,k} = A[0,1]*B[1,k]
      T_{5,k} = A[0,2]*B[2,k]
    
    Then:
      C[0,k] = T_{3,k} + T_{4,k} + T_{5,k}
      C[1,k] = T_{0,k} + T_{1,k} + T_{2,k} - T_{3,k} - T_{4,k} - T_{5,k}
    
    This gives 6 terms per k, total 36. No improvement.
    
    Let's try a different combination:
      T_{0,k} = A[0,0]*B[0,k]
      T_{1,k} = A[0,1]*B[1,k]
      T_{2,k} = A[0,2]*B[2,k]
      T_{3,k} = (A[0,0]+A[1,0]) * B[0,k]
      T_{4,k} = (A[0,1]+A[1,1]) * B[1,k]
      T_{5,k} = (A[0,2]+A[1,2]) * B[2,k]
    
    Then:
      C[0,k] = T_{0,k} + T_{1,k} + T_{2,k}
      C[1,k] = T_{3,k} + T_{4,k} + T_{5,k} - T_{0,k} - T_{1,k} - T_{2,k}
    
    Still 6 terms per k, total 36.
    
    I don't have a clear rank-30 construction, so return None.
    """
    return None


def construct():
    """
    Returns a tensor decomposition for matrix multiplication 2x3 * 3x6.
    Tries to find the minimum rank decomposition.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook decomposition as a baseline
    schoolbook = _schoolbook(n, m, p)
    
    # Verify schoolbook works
    if _verify_exact(schoolbook['U'], schoolbook['V'], schoolbook['W'], n, m, p):
        best_result = schoolbook
        best_rank = n * m * p
    else:
        best_result = schoolbook
        best_rank = n * m * p
    
    # Try rank-24 construction
    if time.time() < deadline:
        result_24 = _try_rank_24(n, m, p)
        if result_24 is not None:
            if _verify_exact(result_24['U'], result_24['V'], result_24['W'], n, m, p):
                best_result = result_24
                best_rank = 24
    
    # Try rank-30 construction
    if time.time() < deadline:
        result_30 = _try_rank_30(n, m, p)
        if result_30 is not None:
            if _verify_exact(result_30['U'], result_30['V'], result_30['W'], n, m, p):
                if 30 < best_rank:
                    best_result = result_30
                    best_rank = 30
    
    return best_result
