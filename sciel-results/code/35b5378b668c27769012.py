"""Program for finding a rank-minimal exact tensor decomposition for 2x4*4x5 matrix multiplication."""
import numpy as np
import random
import time
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    The tensor T has indices (i, j, k) where i in [0,1], j in [0,3], k in [0,4].
    T[i, j, k] = 1 for all i, j, k (standard matrix multiplication tensor).
    
    We use a Kronecker product approach:
    2x4 * 4x5 can be decomposed as:
    - Split the middle dimension m=4 into two blocks of size 2
    - This gives two independent 2x2 * 2x5 multiplications
    - Each 2x2 * 2x5 can be done with rank 10 (schoolbook) or potentially less
    
    However, a better approach: use the fact that the tensor is separable.
    The 2x4*4x5 tensor = (2x2 tensor) ⊗ (2x5 tensor) in some sense.
    
    Actually, let's think more carefully. The standard tensor for matrix multiplication
    A (n x m) * B (m x p) = C (n x p) has T[i,j,k] = 1 for all i,j,k.
    
    For 2x4*4x5: n=2, m=4, p=5, so T is 2x4x5 = 40 entries, all 1s.
    
    The rank of the all-ones tensor of shape (n,m,p) is known to be related to 
    the matrix multiplication tensor. For the standard matrix multiplication tensor,
    the rank is what we're trying to minimize.
    
    Key insight: The tensor T[i,j,k] = 1 can be factored as:
    T[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    where W[k*n+i, c] follows the trace convention.
    
    For the all-ones tensor, we can use:
    - If we set U[i,j,c] = 1, V[j,k,c] = 1, W[k,i,c] = 1 for one c, we get rank 1.
    But this would mean T[i,j,k] = 1*1*1 = 1, which is correct!
    
    Wait, let me verify with the schoolbook decomposition in the seed:
    U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For a specific (i,j,k), only the column c corresponding to (i,j,k) has all three
    non-zero, so T[i,j,k] = 1*1*1 = 1. For other (i',j',k'), the column for (i,j,k)
    doesn't contribute because U[i'*m+j', c] = 0 or V[j'*p+k', c] = 0 or W[k'*n+i', c] = 0.
    
    So the schoolbook gives T[i,j,k] = 1 for all (i,j,k), which is correct.
    
    But can we do better? If T[i,j,k] = 1 for all i,j,k, then rank 1 should work:
    U[i*m+j, 0] = 1 for all i,j
    V[j*p+k, 0] = 1 for all j,k  
    W[k*n+i, 0] = 1 for all k,i
    Then T[i,j,k] = U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1*1*1 = 1. ✓
    
    So rank 1 is sufficient! The answer is rank 1.
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # The tensor T[i,j,k] = 1 for all i,j,k can be decomposed with rank 1
    rank = 1
    
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
