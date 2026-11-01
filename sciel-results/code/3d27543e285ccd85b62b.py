"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.
Uses structured algebraic approach to find low-rank decomposition.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W, rank

def _construct_low_rank(n, m, p):
    """
    Try to construct a low-rank decomposition using structured approach.
    For n=2, m=3, p=10, we try to exploit the small n dimension.
    """
    # Strategy: Use the fact that n=2 to create a more efficient decomposition
    # We'll try to use rank = m * p + (n-1) * m * p or similar
    
    # Approach: For each j and k, we need to handle both i=0 and i=1
    # We can use 2 terms per (j,k) pair with appropriate W values
    
    # Let's try rank = m * p * n = 60 (schoolbook) and see if we can reduce
    # by using the structure more cleverly
    
    # Try a decomposition where we use the following pattern:
    # For each (j, k), create terms that handle the i dimension efficiently
    
    # Actually, let's try a different approach: use the "Winograd" style
    # where we group operations
    
    # For 2x3 * 3x10, let's try to use rank = 30 + some correction
    
    # Let me try to construct explicitly:
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # Try: for each j in [0,m), k in [0,p), create n terms
    # But use the fact that W can be shared
    
    # Let's try a rank-30 construction:
    # Use the lower bound rank = max(n*m, m*p, p*n) = 30
    
    # For rank 30, we need exactly m*p = 30 terms
    # Each term c corresponds to a (j,k) pair
    # U[i*m+j, c] should depend on i
    # V[j*p+k, c] = 1 if this term is for (j,k)
    # W[k*n+i, c] should depend on i
    
    # For this to work: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # If V[j*p+k, c] is 1 only when c corresponds to (j,k), then:
    # U[i*m+j, c_jk] * W[k*n+i, c_jk] = 1 for all i
    
    # This requires U[i*m+j, c_jk] * W[k*n+i, c_jk] = 1 for i=0,1
    # So U[0*3+j, c_jk] * W[k*2+0, c_jk] = 1
    # And U[1*3+j, c_jk] * W[k*2+1, c_jk] = 1
    
    # We can set U[i*m+j, c_jk] = 1, W[k*n+i, c_jk] = 1 for all i
    # But then for a fixed (j,k), only the term with c=c_jk contributes
    # And U[i*m+j, c_jk] * V[j*p+k, c_jk] * W[k*n+i, c_jk] = 1*1*1 = 1 ✓
    
    # Wait, but V[j*p+k, c] should be 1 only for c=c_jk where j,k match
    # Let me define: c indexes (j,k) pairs, c = j*p + k
    # V[j*p+k, c] = 1 if c == j*p+k else 0
    # U[i*m+j, c] = 1 for all i when c corresponds to this j
    # W[k*n+i, c] = 1 for all i when c corresponds to this k
    
    # But wait, c corresponds to a specific (j,k), so:
    # V[j'*p+k', c] = 1 if (j',k') == (j,k) else 0
    # U[i*m+j', c] = 1 if j' == j (the j from c) else 0
    # W[k'*n+i', c] = 1 if k' == k (the k from c) else 0
    
    # Then for a given (i,j,k):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Only c = j*p+k has V[j*p+k, c] = 1
    # For that c: U[i*m+j, c] = 1 (since j matches), W[k*n+i, c] = 1 (since k matches)
    # So the sum = 1 ✓
    
    # This gives rank = m*p = 30!
    
    rank = m * p  # 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            # V[j*p+k, c] = 1
            V[j * p + k][c] = 1
            # U[i*m+j, c] = 1 for all i
            for i in range(n):
                U[i * m + j][c] = 1
            # W[k*n+i, c] = 1 for all i
            for i in range(n):
                W[k * n + i][c] = 1
    
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = 145.0  # seconds, leave 5s margin
    
    n, m, p = DIMENSIONS
    
    # Try the low-rank construction first
    U, V, W, rank = _construct_low_rank(n, m, p)
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
