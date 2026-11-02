"""Tensor decomposition search for 3x3*3x7 matrix multiplication."""
import numpy as np
import time
from itertools import product
from fractions import Fraction
import random

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # The tensor T[i,j,k] = 1 if we multiply A[i,j]*B[j,k]
    # We want to find U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that for all i,j,k: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Strategy: Use the block structure. The 3x3x7 multiplication can be seen as:
    # - Multiplying 3x3 by 3x3 (columns 0-2 of B) -> rank ~23 (Strassen-like)
    # - Multiplying 3x3 by 3x3 (columns 3-5 of B) -> rank ~23
    # - Multiplying 3x3 by 3x1 (column 6 of B) -> rank ~9
    # Total naive: 23+23+9 = 55, but schoolbook is 63.
    # We can try to share some structure.
    
    # Actually, let's think about this differently. The tensor is:
    # T[i,j,k] = 1 for all i in 0..2, j in 0..2, k in 0..6
    # This is a 3x3x7 tensor with all entries 1.
    
    # The rank of this tensor: we need to decompose it as sum of rank-1 tensors.
    # A rank-1 tensor has the form u_i * v_j * w_k.
    # So we need: 1 = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Let's try a randomized approach with small integer/rational entries.
    # We'll search for a decomposition of rank r.
    
    # First, let's try a structured approach based on the all-ones tensor.
    # For the all-ones tensor of size n x m x p, we can use:
    # U[i*m+j, c] = a[i,j,c], V[j*p+k, c] = b[j,k,c], W[k*n+i, c] = w[k,i,c]
    # such that sum_c a[i,j,c]*b[j,k,c]*w[k,i,c] = 1
    
    # One simple approach: use the fact that 1 = 1*1*1, so we can set
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 for one term.
    # But we need sum_c, not just one term. Actually, if we set only c=0 to 1 and rest to 0,
    # then sum = 1*1*1 = 1. So rank 1 would work? No, that's not right.
    # Wait, the tensor entry is T[i,j,k] = 1 for all i,j,k. If we use a single rank-1 term
    # with U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1, then the product is 1 for all i,j,k.
    # So the rank is 1?!
    
    # Let me re-read the problem. "Matrix multiplication 3,3,7" - this is the tensor for
    # multiplying a 3x3 matrix by a 3x7 matrix. The tensor is T[i,j,k] = 1 if the (i,k) entry
    # of the product involves A[i,j]*B[j,k]. Yes, that's the standard matrix multiplication tensor.
    
    # So T[i,j,k] = 1 for all i in 0..2, j in 0..2, k in 0..6. This is indeed the all-ones tensor.
    # And as I noted, a single rank-1 term with all entries 1 gives exactly the all-ones tensor.
    # So the rank is 1!
    
    # Wait, that seems too easy. Let me double-check.
    # U has shape (n*m, rank) = (9, 1), V has shape (m*p, rank) = (21, 1), W has shape (p*n, rank) = (21, 1).
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k
    # W[k*n+i, 0] = 1 for all k,i
    # Then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1 = T[i,j,k]. Yes!
    
    # So the rank-minimal decomposition has rank 1.
    
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
