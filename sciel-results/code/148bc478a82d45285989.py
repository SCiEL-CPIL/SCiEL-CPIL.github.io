"""Matrix multiplication 2x3x8 rank-minimal decomposition search."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# Deadline guard
_DEADLINE = 115  # seconds; evaluator kills at 150s, leave margin

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

_START_TIME = time.time()

# EVOLVE-BLOCK-START

def construct():
    """Find a low-rank exact tensor decomposition for 2x3x8 matrix multiplication."""
    n, m, p = DIMENSIONS
    # Tensor T[i,j,k] = 1 for all i,j,k (it's the full multiplication tensor)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (24, rank)  
    # W shape: (p*n, rank) = (16, rank)
    
    # Strategy: The tensor is the "all-ones" tensor of shape (2,3,8).
    # This is the matmul tensor M_{2,3,8} where C[i,k] = sum_j A[i,j]*B[j,k].
    # In tensor form: T[i,j,k] = 1 (each (i,j,k) triple contributes exactly 1).
    # We need to decompose this as sum of rank-1 tensors.
    
    # Known: schoolbook rank = n*m*p = 48.
    # But we can do better by exploiting structure.
    # 
    # Key insight: For fixed j, the terms involving B[j,k] are shared across i.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Let's use a decomposition based on grouping.
    # For each j (3 values) and k (8 values), we need to multiply A[i,j] * B[j,k]
    # and accumulate into C[i,k].
    #
    # A standard approach: For each (j,k) pair, we have 2 values of i.
    # We can write: for each j, use a "fan-in" structure.
    #
    # Actually, let me think about this differently.
    # The tensor T is of size 2*3*8 = 48, and T[i,j,k]=1 for all i,j,k.
    # This is the all-ones tensor J_{2,3,8}.
    #
    # Rank of all-ones tensor J_{a,b,c} is max(a,b,c) when one dimension is 1... 
    # No, that's not right. The all-ones tensor has rank 1!
    # Wait: J[i,j,k] = 1 = 1*1*1, so it's rank 1!
    #
    # But that's the tensor where every entry is 1. Is our matmul tensor really all-ones?
    # Yes! T[i,j,k] = 1 for all valid i,j,k. Each triple (i,j,k) has coefficient 1.
    #
    # So T = 1_i * 1_j * 1_k is rank 1!
    #
    # Wait, that can't be right for matrix multiplication. Let me reconsider.
    #
    # The matrix multiplication tensor for C = A*B where A is 2x3, B is 3x8:
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    #
    # As a trilinear form: sum_{i,j,k} C[i,k] * A[i,j] * B[j,k] with coefficient 1.
    # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,7}.
    #
    # So T is indeed the all-ones tensor of shape (2,3,8), which has rank 1!
    #
    # Rank-1 decomposition:
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k
    # W[k*n+i, 0] = 1 for all k,i
    #
    # Then sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1*1*1 = 1 = T[i,j,k]. 
    #
    # So the rank is 1!
    
    rank = 1
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for idx in range(n * m):
            U[idx][r] = 1
        for idx in range(m * p):
            V[idx][r] = 1
        for idx in range(p * n):
            W[idx][r] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
