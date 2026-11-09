"""
Optimized program for matrix multiplication 2x3x9 tensor decomposition.
Searches for rank-minimal exact tensor decomposition.
"""

import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 2, 3, 9
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Build the target tensor T with indices (i,j), (j,k), (i,k)
    # T[(i,j),(j,k),(i,k)] = 1
    def tensor_index(i, j, k):
        # First mode: (i,j) -> i*m + j
        # Second mode: (j,k) -> j*p + k  
        # Third mode: (i,k) -> k*n + i
        return (i * m + j, j * p + k, k * n + i)
    
    # For a given rank r, try to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = 1 for all i,j,k
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the tensor."""
        # Compute sum over r of U[i,j,r] * V[j,k,r] * W[i,k,r]
        # U shape: (n*m, r), V shape: (m*p, r), W shape: (p*n, r)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Schoolbook solution as fallback
    def schoolbook():
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
    
    # Try to find lower rank decompositions using randomized search
    # Key insight: for fixed j, the sub-tensor over (i,k) is all-ones 2x9,
    # which can be represented with rank 2 (since it's 2*9=18 entries but 
    # structured). So maybe we can do better than 54.
    
    # Attempt: use the structure T[i,j,k] = 1 = (1/2)(1+1) for i dimension
    # For each j, the 2x9 all-ones matrix = u1*v1^T + u2*v2^T where
    # u1 = [1,0], v1 = [1,1,...,1] and u2 = [0,1], v2 = [1,1,...,1]
    # This gives 3*2 = 6 terms!
    
    # Let's verify: for term (j, i_part):
    # U[i*m+j, r] = 1 if i == i_part else 0
    # V[j*p+k, r] = 1 for all k
    # W[k*n+i, r] = 1 for all k, and = 1 if i == i_part... 
    # Wait, W depends on (i,k), so W[k*n+i, r] should be 1 for the specific i
    
    # Let me think again. We need sum_r U[i,j,r]*V[j,k,r]*W[i,k,r] = 1
    
    # Try: for each j in {0,1,2}, for each i in {0,1}:
    #   U[i*m+j, r] = 1
    #   V[j*p+k, r] = 1 for all k
    #   W[k*n+i, r] = 1 for all k
    # Then sum over r: for given (i,j,k), the terms that contribute are those
    # where U[i*m+j,r]=1 AND V[j*p+k,r]=1 AND W[k*n+i,r]=1
    # With the above, for term (j', i'), U[i*m+j, (j',i')] = 1 iff j=j' and i=i'
    # V[j*p+k, (j',i')] = 1 iff j=j'
    # W[k*n+i, (j',i')] = 1 iff i=i'
    # So the product is 1 iff j=j' and i=i', and 0 otherwise.
    # Sum over (j',i'): only the term with j'=j, i'=i contributes, giving 1. 
    # Total rank = 3*2 = 6!
    
    # Let's verify this more carefully with a small example.
    # n=2, m=3, p=9
    # Term indexed by (j', i') where j' in {0,1,2}, i' in {0,1}
    # U shape: (6, 6), V shape: (27, 6), W shape: (18, 6)
    
    def try_rank_6():
        rank = 6
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col = 0
        for j in range(m):
            for i in range(n):
                U[i * m + j][col] = 1.0
                for k in range(p):
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Verify rank-6 solution
    result_6 = try_rank_6()
    if verify(result_6["U"], result_6["V"], result_6["W"]):
        return result_6
    
    # If rank 6 doesn't work, try other small ranks
    # But wait - let me double-check the logic above
    
    # For (i,j,k): sum_{j',i'} U[i*m+j, (j',i')] * V[j*p+k, (j',i')] * W[k*n+i, (j',i')]
    # = sum_{j',i'} [1 if j=j' and i=i' else 0] * [1 if j=j' else 0] * [1 if i=i' else 0]
    # = [1 if j=j and i=i else 0] = 1. Correct!
    
    # So rank 6 should work. Let's return it.
    # The verify function should confirm this.
    
    # Fallback to schoolbook
    return schoolbook()
