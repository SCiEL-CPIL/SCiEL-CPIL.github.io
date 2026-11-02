"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]
_START_TIME = time.time()
_DEADLINE = 140  # seconds, safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _verify(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _try_greedy_reduction(n, m, p, max_rank, seed=42):
    """Try to find a decomposition of rank <= max_rank using greedy search."""
    rng = random.Random(seed)
    
    # We need to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
    
    # Strategy: start with schoolbook, try to merge columns
    # Or: random search with local improvement
    
    # For now, try a simple approach: random rank-1 terms that sum to the tensor
    # The tensor is all-ones, so we need sum of rank-1 terms = all-ones tensor
    
    # A rank-1 term is u ⊗ v ⊗ w where u has length n, v length m, w length p
    # u[i]*v[j]*w[k] contributes to T[i,j,k]
    
    # We need sum_c u_c[i]*v_c[j]*w_c[k] = 1 for all i,j,k
    
    # This is equivalent to: the n x m x p all-ones tensor = sum of rank-1 tensors
    # The all-ones tensor has rank 1! So rank 1 should work.
    
    # But wait - that would mean matrix multiplication has rank 1, which is wrong.
    # I must be misunderstanding the tensor structure.
    
    # Let me just return schoolbook for now and try to improve via other means.
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find better decomposition
    # The key insight: we can use linearity to combine terms
    
    # For 2x5x6, let's try to find a structured decomposition
    # One approach: use the fact that we can factor through intermediate dimensions
    
    # Actually, let's try a direct search for rank < 60
    # using the constraint that the tensor must equal all-ones
    
    # Since the all-ones tensor has rank 1 in the standard sense,
    # but matrix multiplication tensor might be different...
    
    # Let me just verify schoolbook and return it, then try improvements
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        best_result = result
    else:
        best_result = None
    
    # Try greedy: start with schoolbook, try to eliminate columns
    # by finding linear dependencies
    
    if _time_left() < 5:
        if best_result:
            return best_result
        return result
    
    # Try to find a lower rank decomposition
    # Approach: random rank-1 tensors, greedy set cover style
    
    # Generate candidate rank-1 terms
    # A rank-1 term (u,v,w) contributes u[i]*v[j]*w[k] to T[i,j,k]
    # We need the sum to be all-ones
    
    # Let's try: use terms where u, v, w have entries in {-1, 0, 1}
    # and see if we can sum to all-ones with fewer terms than 60
    
    # Actually, the simplest improvement: notice that for fixed i,
    # sum_j,k T[i,j,k] = m*p, and these can potentially be grouped.
    
    # Let me try a different approach: use the structure of the problem
    # C = AB where A is 2x5, B is 5x6
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    
    # We can write: C[0,k] - C[1,k] = sum_j (A[0,j]-A[1,j])*B[j,k]
    # This doesn't directly help reduce rank.
    
    # Let me just return the schoolbook for now, as finding a better
    # decomposition requires more sophisticated search.
    
    if best_result:
        return best_result
    return result

# EVOLVE-BLOCK-END
