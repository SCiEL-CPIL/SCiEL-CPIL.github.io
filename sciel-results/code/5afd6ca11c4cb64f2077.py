"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    # We want to express this as sum of rank-1 tensors
    
    # Strategy: Use the fact that for fixed i and k, the j dimension 
    # contributes to a specific pattern. We can try to share factors.
    
    # For 2x4x7, let's try rank 49 or lower
    # Actually, let me try a direct construction
    
    # The multiplication can be viewed as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # A rank-1 decomposition term contributes:
    # U[a, r] * V[b, r] * W[c, r] where a=i*m+j, b=j*p+k, c=k*n+i
    
    # Let me try to find a decomposition with rank < 56
    # by using linearity in the middle dimension
    
    # For each (i,k) pair, we need sum_j A[i,j]*B[j,k]
    # This is like a bilinear form in j
    
    # Key insight: If we can find a low-rank factorization of the 
    # "connection" between A and B for each (i,k), we can reduce rank.
    
    # Let me try rank = 49 (7*7) or other values
    
    # Actually, let me try a different approach: use the structure that
    # m=4=2*2, so we can split the middle dimension
    
    # For now, return schoolbook if we can't find better quickly
    # But let me try a simple optimization: merge terms where possible
    
    # Try rank from 40 to 56 and see if we can construct one
    # Given time constraints, let me just return schoolbook for now
    # unless I can quickly find a better one
    
    deadline = time.time() + 140  # 10s safety margin
    
    # Let me try a construction with rank = 49
    # Idea: For each k (0..6) and each i (0..1), we have a 4-dim dot product
    # If we can express each 4-dim dot product with rank < 4 on average...
    
    # Actually, the 4-dim dot product has rank 4 (it's just sum of 4 products)
    # So total rank would be 2*7*4 = 56 (schoolbook)
    
    # To reduce rank, we need to share across different (i,k) pairs
    # This is possible if the patterns overlap
    
    # Let me try: for fixed j, the term A[i,j]*B[j,k] appears for all i,k
    # So if we could factor A[i,j] and B[j,k] independently...
    
    # But that's exactly what schoolbook does: each (i,j,k) is separate
    
    # Alternative: use the fact that we can combine terms
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # This is a sum of 4 rank-1 terms in (A_slice, B_slice) for each (i,k)
    # Total: 2*7*4 = 56 rank-1 terms
    
    # Can we do better? The tensor is:
    # T[i,j,k] = delta_{j,k} in the appropriate sense
    
    # Let me just return schoolbook for safety, as finding a provably 
    # lower rank decomposition for 2x4x7 requires more sophisticated methods
    
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
