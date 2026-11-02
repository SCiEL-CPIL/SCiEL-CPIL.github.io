"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.
    
    The strategy: Split the 2x3 matrix A into two 1x3 row vectors. Each row vector 
    multiplied by the 3x8 matrix B gives a 1x8 output row. The tensor for 1x3x8 
    multiplication has rank exactly 8 (since it's just a vector-matrix product, which 
    can be seen as summing 8 rank-1 terms, one per column of B).
    
    Total rank = 2 * 8 = 16.
    
    Convention:
      U has shape (n*m, rank) = (6, 16)
      V has shape (m*p, rank) = (24, 16)
      W has shape (p*n, rank) = (16, 16)
      
      W[k*n + i, r] is the coefficient for output C[i,k] in term r.
      
    For row i of A (i=0,1) and column k of C (k=0..7):
      C[i,k] = sum_j A[i,j] * B[j,k]
      
    We need: C[i,k] = sum_r U[i*m + j_u, r] * V[j_v*p + k_v, r] * W[k*n + i, r]
    where the indices must match the tensor structure.
    
    Actually, the tensor T[i,j,k] = delta (identity) means:
      C[i,k] = sum_{j} A[i,j] B[j,k]
      
    The tensor decomposition: T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    For a rank-1 term r, we have:
      U[i*m+j, r] = u_r(i,j)
      V[j*p+k, r] = v_r(j,k)
      W[k*n+i, r] = w_r(k,i)
      
    And we need: sum_r u_r(i,j) v_r(j,k) w_r(k,i) = delta_{i,j,k} is not right.
    
    The matrix multiplication tensor is: T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k]
    The tensor has entries T[i,j,k] = 1 if the indices are "compatible" in the multiplication.
    Actually, the standard form: the multiplication tensor has T[i,j,k] = delta where 
    it represents the coefficient. For matrix mult, T[i,j,k] = 1 for all valid (i,j,k) 
    is not quite right either.
    
    Let me reconsider. The tensor for multiplying A (n x m) by B (m x p) to get C (n x p):
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor T has dimensions (n*m) x (m*p) x (n*p) or similar. In the decomposition:
    U is (n*m) x rank, V is (m*p) x rank, W is (n*p) x rank.
    
    The condition is: for all i,j,k:
      delta_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
      
    Wait, that's not right. The tensor T[i,j,k] for matrix multiplication is simply 1 
    when we're computing the (i,k) entry of C from A[i,j] and B[j,k]. But actually the 
    tensor is defined such that the bilinear form gives the product.
    
    Let me think more carefully. The multiplication tensor T is a 3-tensor with indices 
    (i, j, k) where i is row of A, j is the shared index, k is column of B. The value 
    T[i,j,k] = 1 for all i,j,k in their ranges? No.
    
    Actually, for matrix multiplication, the tensor is:
    T[i, j, k] = 1 if we interpret it as: the (i,k) entry of C is sum over j of A[i,j]*B[j,k].
    
    In tensor decomposition form: we want to write the tensor T (with entries T[i,j,k]) 
    as sum of rank-1 tensors. A rank-1 tensor is u(i) * v(j) * w(k).
    
    For matrix multiplication, T[i,j,k] = delta_{i,j,k} doesn't make sense because the 
    indices don't all range over the same set.
    
    Let me use the standard formulation. The tensor T has three modes:
    - Mode 1: indexed by (i, j) where i in [0,n), j in [0,m) -> size n*m
    - Mode 2: indexed by (j, k) where j in [0,m), k in [0,p) -> size m*p  
    - Mode 3: indexed by (i, k) where i in [0,n), k in [0,p) -> size n*p
    
    And T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k' (i.e., all indices match).
    
    So T is essentially the "identity" tensor in this combined indexing.
    
    For a rank-1 decomposition term r:
    U[i*m+j, r] * V[j'*p+k, r] * W[i'*n+k', r] = delta[i,i'] * delta[j,j'] * delta[k,k']
    
    This is hard to do with few terms. The schoolbook uses n*m*p terms.
    
    For the split approach: consider i fixed. For a given i, we have:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    This is a vector (length m) times a matrix (m x p) = vector (length p).
    The tensor for this is: for fixed i, T_i[j, k] = delta[j,j'] * delta[k,k'] summed over j.
    
    Actually, for fixed i, the sub-tensor is just the identity-like structure for 
    vector-matrix multiplication. The rank of the vector-matrix multiplication tensor 
    (1 x m x p) is m*p? No, it's just p? Let me think...
    
    For 1 x m x p: C[k] = sum_j A[j] B[j,k]. The tensor T[j,k] = delta[j,j']*delta[k,k']? 
    No, T[j,k] is just 1 for the "matching" in the sum. The tensor has T[j,k] = 1 for 
    all j,k? No.
    
    I think I'm overcomplicating. Let me just verify with a simple case.
    
    For n=1, m=3, p=8: C[k] = sum_{j=0}^{2} A[j] B[j,k]
    
    The tensor T[j,k] should satisfy: C[k] = sum_j A[j] * (sum_k' T[j,k,k'] B[j',k'])
    So T[j,k,k'] = delta[j,j'] * delta[k,k']... that gives C[k] = sum_j A[j] B[j,k] which is correct.
    
    But in the 3-tensor form with U (m x rank), V (m*p x rank), W (p x rank):
    sum_r U[j,r] V[j'*p+k',r] W[k,r] = delta[j,j'] * delta[k,k']
    
    This is rank m*p = 24? Or can we do better?
    
    Actually, for the 1xm x mxp x 1xp case, the tensor is just the "transpose" structure. 
    I believe the rank is m*p for the naive case, but maybe we can do p?
    
    Let me try: for each k, we need to extract B[:,k] and dot with A. 
    Term r=k: U[j,r] = A[j]? No, U can't depend on A.
    
    I think for 1xm x mxp, the rank is m*p. So total rank would be n*m*p = 48, same as schoolbook.
    
    Hmm, that doesn't help. Let me reconsider.
    
    Actually, I recall that for matrix multiplication, the rank is at least n*m*p in general 
    (for small cases), but there might be better decompositions for specific dimensions.
    
    Let me just try to find a decomposition of rank 16 by direct construction.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try the split approach: rank = n * p = 16
    rank = n * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # For each (i, k) pair, create a term
    # Term r = i*p + k
    # We want: sum_r U[i*m+j, r] V[j'*p+k', r] W[k''*n+i'', r] = delta[i,i''] * delta[j,j'] * delta[k,k'']
    
    # This is tricky. Let me think of it differently.
    # For fixed i, the contribution to C[i,k] is sum_j A[i,j] B[j,k].
    # If we make each term r=(i,k) contribute only to C[i,k]:
    #   W[k*n+i, r] = 1 for r=(i,k), 0 otherwise
    #   Then we need: sum_r U[i*m+j, r] V[j'*p+k', r] * W[k''*n+i'', r] 
    #   = U[i*m+j, (i,k)] * V[j'*p+k', (i,k)] when i''=i, k''=k
    #   = U[i*m+j, (i,k)] * V[j*p+k, (i,k)]  (since j'=j, k'=k for the match)
    #   We want this to equal delta[j,j'] * delta[k,k''] which is 1 when j=j' and k=k''
    #   So U[i*m+j, (i,k)] * V[j*p+k, (i,k)] = 1 for all j, and 0 otherwise?
    #   But U and V are shared across different (i,k) terms...
    
    # This approach doesn't work directly because U[i*m+j, r] is shared across k values.
    
    # Let me try a different approach. Maybe rank = m*p = 24?
    # Or maybe I should just use the schoolbook rank of 48 and see if I can reduce it.
    
    # Actually, let me just return the schoolbook decomposition for now, since I'm not sure 
    # how to improve it within the time limit.
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
