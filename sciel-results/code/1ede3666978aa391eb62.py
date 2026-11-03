import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Build a rank-minimal exact decomposition for 2x3x10 matrix multiplication.
    # Strategy:
    # - For each (i, j) pair (i in 0..n-1, j in 0..m-1), the row i*m+j of U and 
    #   the row j*p+k of V for each k, and the row k*n+i of W must satisfy:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r]  ... wait, no.
    #   
    #   Actually, the tensor decomposition condition is:
    #   For all i, j, k: 
    #     sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  (if (i,j,k) is a "valid" triple)
    #     sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 0  (otherwise)
    #   
    #   Wait, the standard matrix multiplication tensor T[i,j,k] = 1 if we're multiplying 
    #   A[i,j] * B[j,k] to get C[i,k]. The tensor is T_{i,j,k} = 1 for all i,j,k.
    #   Actually no: the tensor for matrix multiplication C = A * B is:
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    #   So the tensor T[i,j,k] = 1 for all i, j, k (where i indexes rows of A, j indexes 
    #   columns of A / rows of B, k indexes columns of B).
    #   
    #   The rank decomposition means:
    #   T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #   
    #   Wait, that's not right either. Let me re-read the task.
    #   
    #   "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix 
    #    by a 3 by 10 matrix. The output factor follows the official trace convention W[k*n+i,r]."
    #   
    #   The matrix multiplication tensor for (n,m,p) has entries T[i,j,k] = 1 for all 
    #   i in [0,n), j in [0,m), k in [0,p). The rank decomposition is:
    #   T[i,j,k] = sum_{r=0}^{R-1} U[i,j,r] * V[j,k,r] * W[k,i,r]
    #   where U has shape (n*m, R), V has shape (m*p, R), W has shape (p*n, R).
    #   
    #   With the index conventions:
    #   U row index = i*m + j
    #   V row index = j*p + k
    #   W row index = k*n + i
    #   
    #   So we need: for all i, j, k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #   
    #   The trivial decomposition uses R = n*m*p = 60, with one term per (i,j,k) triple.
    #   
    #   To reduce rank, we can use the fact that for fixed i and j, the values V[j*p+k, r] 
    #   and W[k*n+i, r] vary with k. If we can share terms across different k values, we 
    #   reduce the rank.
    #   
    #   One approach: for each (i, j) pair, use a small number of terms that cover all k.
    #   For example, if for a given (i, j), we use R_ij terms such that for each k, 
    #   sum_{r in terms for (i,j)} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
    #   
    #   A simple way: for each (i, j), pick one term r_{ij} and set:
    #   U[i*m+j, r_{ij}] = 1
    #   V[j*p+k, r_{ij}] = 1 for all k
    #   W[k*n+i, r_{ij}] = 1 for all k
    #   Then sum_r ... = 1 for all k. This uses n*m = 6 terms total, but wait, that would 
    #   mean V[j*p+k, r_{ij}] = 1 for all k, which is fine. And W[k*n+i, r_{ij}] = 1 for 
    #   all k. Let's check: for a fixed (i,j,k), the sum over r is just the term r_{ij}, 
    #   giving U[i*m+j, r_{ij}] * V[j*p+k, r_{ij}] * W[k*n+i, r_{ij}] = 1 * 1 * 1 = 1. 
    #   For other (i',j',k') triples, we need the sum to be 0. But if we use distinct r_{ij} 
    #   for each (i,j), then for a triple (i',j',k') where (i',j') != (i,j), the term r_{ij} 
    #   has U[i'*m+j', r_{ij}] = 0 (since we only set U[i*m+j, r_{ij}] = 1), so the product 
    #   is 0. So this works with rank = n*m = 6!
    #   
    #   Wait, let me double-check. We have n*m = 6 distinct rank indices, one for each 
    #   (i,j) pair. For rank index r_{ij}:
    #   - U[i*m+j, r_{ij}] = 1, and U[i'*m+j', r_{ij}] = 0 for (i',j') != (i,j).
    #   - V[j*p+k, r_{ij}] = 1 for all k.
    #   - W[k*n+i, r_{ij}] = 1 for all k.
    #   
    #   Now, for a given (i,j,k), the sum is over all r. Only r = r_{ij} contributes, 
    #   giving 1 * 1 * 1 = 1. Correct.
    #   
    #   For a different triple (i',j',k') where (i',j') != (i,j): the only potentially 
    #   non-zero term is r = r_{i'j'}, but U[i*m+j, r_{i'j'}] = 0 (since (i,j) != (i',j')), 
    #   so the product is 0. Wait, no: we're evaluating T[i',j',k'] = sum_r U[i'*m+j', r] * 
    #   V[j'*p+k', r] * W[k'*n+i', r]. The term r = r_{i'j'} gives 1 * 1 * 1 = 1. All 
    #   other terms r = r_{ij} with (i,j) != (i',j') give U[i'*m+j', r_{ij}] = 0. So 
    #   T[i',j',k'] = 1. Correct.
    #   
    #   So the rank is n*m = 6. This is much better than 60!
    #   
    #   Let's verify with a small example. n=1, m=1, p=1. Rank = 1. 
    #   U[0,0] = 1, V[0,0] = 1, W[0,0] = 1. T[0,0,0] = 1. Correct.
    #   
    #   n=1, m=2, p=1. Rank = 2.
    #   r_{0,0} = 0, r_{0,1} = 1.
    #   U[0,0]=1, U[1,1]=1.
    #   V[0,0]=1, V[0,1]=1 (j=0, k=0).
    #   V[1,0]=1, V[1,1]=1 (j=1, k=0).
    #   W[0,0]=1, W[0,1]=1 (k=0, i=0).
    #   T[0,0,0] = U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] = 1*1*1 + 0*1*1 = 1. Correct.
    #   T[0,1,0] = U[1,0]*V[1,0]*W[0,0] + U[1,1]*V[1,1]*W[0,1] = 0*1*1 + 1*1*1 = 1. Correct.
    #   
    #   Great, this works! The rank is n*m = 6 for 2x3x10.
    
    rank = n * m
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j, r] = 1
            for k in range(p):
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
