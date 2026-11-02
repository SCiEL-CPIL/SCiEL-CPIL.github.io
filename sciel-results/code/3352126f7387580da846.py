"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 140  # safety margin
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U, V, W, rank = schoolbook()
    
    def to_list(mat):
        return mat.tolist()
    
    def verify(U, V, W, tol=1e-9):
        """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def try_reduce_rank(U, V, W, current_rank, max_attempts=50):
        """Try to reduce rank by 1 by merging two terms"""
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            # Pick two random columns to try to merge
            a, b = random.sample(range(current_rank), 2)
            # We want to find new U', V', W' for columns a,b such that
            # the combined effect equals the original two columns
            # Original contribution: U[:,a]*V[:,a]*W[:,a] + U[:,b]*V[:,b]*W[:,b]
            # New contribution: U[:,a']*V[:,a']*W[:,a'] (single term)
            # This is generally not possible, but we can try to adjust all terms
            
            # Alternative: try to express column b as a linear combination of others
            # This is more complex. Let's try a different approach.
            
            # Try: set W[k*n+i, b] = 0 for all k,i, and adjust other W entries
            # to compensate. This requires solving a linear system.
            
            # For each (i,j,k), we need:
            # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # If we zero out column b's W, we need to add to other W entries:
            # delta_W[k*n+i] = -U[i*m+j,b]*V[j*p+k,b] / (U[i*m+j,r']*V[j*p+k,r'])
            # This is only possible if the denominator is nonzero and consistent.
            
            # This approach is too complex. Let's just return current rank.
            pass
        return None
    
    # For now, return schoolbook. In a more advanced version, we'd try to reduce.
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
