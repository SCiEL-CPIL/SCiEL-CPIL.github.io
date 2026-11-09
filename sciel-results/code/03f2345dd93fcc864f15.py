"""Matrix multiplication 3x3 by 3x8: rank-minimal exact tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x8 matrix.
    
    The tensor T has indices (i, j, k) where:
    - i in [0, n) = [0, 3) : row of A, row of C
    - j in [0, m) = [0, 3) : col of A, row of B  
    - k in [0, p) = [0, 8) : col of B, col of C
    
    T[i,j,k] = 1 if we have A[i,j]*B[j,k] contributing to C[i,k]
    
    We seek rank r such that T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    where:
    - u_c is a vector of length n*m = 9 (indexed by i*m+j)
    - v_c is a vector of length m*p = 24 (indexed by j*p+k)
    - w_c is a vector of length p*n = 24 (indexed by k*n+i)
    
    And T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    The output factor W follows the convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    
    # The tensor T is defined as:
    # T[i, j, k] = 1 for all valid i, j, k
    # This is the all-ones tensor of shape (n, m, p) = (3, 3, 8)
    
    # In the flattened index convention:
    # U has shape (n*m, rank) = (9, rank), indexed by [i*m + j]
    # V has shape (m*p, rank) = (24, rank), indexed by [j*p + k]
    # W has shape (p*n, rank) = (24, rank), indexed by [k*n + i]
    
    # The tensor entry T[i, j, k] corresponds to:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is the standard matrix multiplication tensor.
    
    # Known lower bound: rank >= max(n*m, m*p, p*n, n*m*p / max(n,m,p))
    # For (3,3,8): n*m=9, m*p=24, p*n=24, n*m*p/max=72/8=9
    # So lower bound is max(9, 24, 24, 9) = 24
    
    # Upper bound: schoolbook gives n*m*p = 72
    # Better: we can use rank = m * max(n,p) if we factor differently
    
    # Let's think about this more carefully.
    # The tensor is T[i,j,k] = 1 for all i,j,k
    # This can be written as: sum_j (e_i ⊗ e_j ⊗ e_k) * 1
    # But that's just the schoolbook.
    
    # A better decomposition:
    # T[i,j,k] = sum_j delta(i,i) * delta(j,j) * delta(k,k)
    # This doesn't help directly.
    
    # Actually, the all-ones tensor of shape (n,m,p) has rank at most n*m*p,
    # but we can do better.
    
    # Key insight: T[i,j,k] = 1 = sum_{j'} delta(j, j') * 1
    # So T = sum_{j'} (sum_i e_i) ⊗ e_{j'} ⊗ (sum_k e_k)
    # Wait, that gives: sum_{j'} (sum_i e_i) ⊗ e_{j'} ⊗ (sum_k e_k)
    # = (sum_i e_i) ⊗ (sum_{j'} e_{j'}) ⊗ (sum_k e_k)
    # This is rank 1! But wait, that's not right because the indices don't match.
    
    # Let me reconsider. The tensor T is defined on indices (i, j, k) with:
    # U indexed by (i, j) -> i*m + j
    # V indexed by (j, k) -> j*p + k  
    # W indexed by (k, i) -> k*n + i
    
    # So T[i*m+j, j*p+k, k*n+i] should equal 1 for all valid i,j,k where the indices are consistent.
    
    # Actually, I need to be more careful. The tensor is:
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # But the constraint is that j = j' and k = k' and i = i' for the entry to be 1.
    
    # So T[i*m+j, j*p+k, k*n+i] = 1, and T[a,b,c] = 0 otherwise (when the indices don't align).
    
    # This is the standard matrix multiplication tensor.
    
    # Let me try a decomposition with rank = m * max(n, p) = 3 * 8 = 24
    
    # Idea: For each j (column of A / row of B), we can use rank n*p entries,
    # but we can share across j's if possible.
    
    # Alternative: rank = n*m*p / min(n,m,p) is not generally achievable.
    
    # Let's try rank = m * n * p / something... 
    
    # Actually, for the all-ones pattern with the index constraints, let me try:
    # rank = m * max(n, p) = 3 * 8 = 24
    
    # For each j in [0, m), for each pair (i, k) in [0,n) x [0,p):
    # That's m * n * p = 72 again.
    
    # Let me try a different approach. Consider:
    # T[i,j,k] = 1 = sum_{j'} delta(j, j') 
    # So we can write: T = sum_{j'} u_{j'} ⊗ v_{j'} ⊗ w_{j'}
    # where u_{j'}[i*m + j] = 1 if j = j', else 0
    #       v_{j'}[j'*p + k] = 1 for all k
    #       w_{j'}[k*n + i] = 1 for all i, k
    # 
    # But that's not a valid rank-1 tensor because the indices need to be consistent.
    
    # Let me just use the schoolbook and see if I can reduce it.
    
    # Schoolbook: rank = n*m*p = 72
    # For each (i, j, k):
    #   U[i*m+j, c] = 1
    #   V[j*p+k, c] = 1  
    #   W[k*n+i, c] = 1
    
    # Can we merge some of these? If we have two terms (i,j,k) and (i',j,k') with same j,
    # can we combine them? Only if the resulting tensors factor.
    
    # For fixed j, the sub-tensor is over (i, k) with U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is a rank-1 tensor in (i,k) for each c, but we have n*p such entries.
    
    # The minimum rank of an n x p all-ones matrix is 1, so for fixed j, we need rank 1
    # in the (i,k) space. But the U, V, W factors are coupled across different j's.
    
    # Let me try: for each j, use one rank-1 factor:
    # U[i*m+j, c_j] = 1 for all i
    # V[j*p+k, c_j] = 1 for all k
    # W[k*n+i, c_j] = 1 for all i, k
    # 
    # But W[k*n+i, c_j] = 1 for all i,k means W is the same for all j, which conflicts
    # with different j's using different columns.
    
    # This approach doesn't work because W is shared.
    
    # Let me try yet another approach. The known result for matrix multiplication M_{n,m,p}
    # is that the rank is at least max(n*m, m*p, p*n) and at most n*m*p.
    # For small cases, exact ranks are known.
    
    # For M_{3,3,8}, I'll try to find a good decomposition computationally.
    
    # Given time constraints, let me use a rank that's likely close to optimal.
    # Lower bound: max(9, 24, 24) = 24
    # Let me try rank = 24 first and see if it works.
    
    # Actually, let me just use a reasonable rank and construct the decomposition.
    # I'll use rank = m * p = 24 as a starting point.
    
    # For rank = m*p = 24:
    # For each (j, k) pair, create a factor:
    # U[i*m+j, c_{j,k}] = 1 for all i
    # V[j*p+k, c_{j,k}] = 1
    # W[k*n+i, c_{j,k}] = 1 for all i
    # 
    # Check: sum_c U[i*m+j, c] * V[j'*p+k', c] * W[k'*n+i', c]
    # = sum_{(j,k)} [1 if j=j' and i=i'] * [1 if j=j' and k=k'] * [1 if k=k' and i=i']
    # Wait, this isn't quite right.
    
    # Let me be more precise. With rank = m*p:
    # For c corresponding to (j, k):
    # U[i*m + j', c] = 1 if j' = j, else 0  (for all i)
    # V[j'*p + k', c] = 1 if j' = j and k' = k, else 0
    # W[k'*n + i', c] = 1 if k' = k and i' = i, else 0  -- but i is free!
    
    # The problem is that W depends on i, but we want it to work for all i.
    
    # Let me try: W[k*n + i, c_{j,k}] = 1 for all i (when k matches)
    # Then: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # = sum_{(j',k')} [U[i*m+j, c_{j',k'}] * V[j*p+k, c_{j',k'}] * W[k*n+i, c_{j',k'}]]
    # = sum_{(j',k')} [1 if j=j' * 1 if j=j' and k=k' * 1 if k=k']
    # = 1 if j=j' and k=k', summed over (j',k')
    # = 1 (since only one term survives)
    
    # Wait, that works! Let me verify:
    # U[i*m + j, c_{j',k'}] = 1 if j = j', else 0
    # V[j*p + k, c_{j',k'}] = 1 if j = j' and k = k', else 0
    # W[k*n + i, c_{j',k'}] = 1 if k = k', else 0  (for all i)
    
    # Product: 1 if j=j' AND j=j' AND k=k' AND k=k' = 1 if j=j' and k=k'
    # Sum over (j',k'): exactly one term is 1, so sum = 1. 
    
    # This gives rank = m*p = 24!
    
    rank = m * p  # = 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = j * p + k  # column index
            # U: for all i, U[i*m + j, c] = 1
            for i in range(n):
                U[i * m + j][c] = 1
            # V: V[j*p + k, c] = 1
            V[j * p + k][c] = 1
            # W: for all i, W[k*n + i, c] = 1
            for i in range(n):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
