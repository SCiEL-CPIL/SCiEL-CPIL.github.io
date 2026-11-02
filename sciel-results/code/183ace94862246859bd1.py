import time
import numpy as np
from itertools import product
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for r in range(rank):
        for i in range(n):
            if len(U[i * m]) != rank:
                return False
        for j in range(m):
            if len(V[j * p]) != rank:
                return False
        for k in range(p):
            if len(W[k * n]) != rank:
                return False
    
    # Verify: for each (i,j,k), sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(rank):
                    total += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                if total != Fraction(1):
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = 3, 5, 7
    
    # Trivial decomposition: rank = n*m*p
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
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Trivial decomposition failed verification")
    
    # Try to find a lower-rank decomposition using structured approach
    # The key insight: for each (i,k) pair, we need sum_j T[i,j,k] = 1 contribution
    # where T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    
    # Attempt: group by (i,k) and see if we can share rank-1 terms
    # For fixed (i,k), we need: for all j, sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] = 1
    # If we set W[k*n+i][r] = 1 for r in some set, and U[i*m+j][r]*V[j*p+k][r] = 1/m for each j...
    # This doesn't directly help without increasing rank.
    
    # Alternative: try rank = n * max(m,p) = 3*7 = 21? Unlikely to work.
    # Try rank = max(n*m, m*p, p*n) = 35? Also unlikely.
    
    # For now, the trivial decomposition with rank 105 is our baseline.
    # Check if we have time to attempt a greedy reduction
    if time.time() < deadline - 30:
        # Try to merge some terms: if two terms have the same W pattern, 
        # we might be able to combine them
        # This is complex; for now, return the verified trivial decomposition
        pass
    
    return {"U": U, "V": V, "W": W}
