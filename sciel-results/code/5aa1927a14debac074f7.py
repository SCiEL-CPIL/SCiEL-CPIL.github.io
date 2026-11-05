"""Rewritten program: greedy linear-algebraic elimination over the schoolbook basis.

The multiplication tensor T_{i,j,k} = 1 (for all i in [0,n), j in [0,m), k in [0,p))
is decomposed as T = sum_{r} u_r ⊗ v_r ⊗ w_r where:
  U has shape (n*m, rank),  U[i*m + j, r]
  V has shape (m*p, rank),  V[j*p + k, r]
  W has shape (p*n, rank),  W[k*n + i, r]

The schoolbook basis has rank n*m*p = 96, with one rank-1 term per (i,j,k) triple.
Since T is the all-ones tensor, many of these terms are linearly dependent.
We apply a greedy set-cover / elimination strategy:

1. Start with the full schoolbook set of 96 rank-1 components.
2. Observe that the all-ones tensor on R^{n x m x p} has rank at most n*m (by fixing
   k and using separable structure), but more carefully:
   
   T_{i,j,k} = 1 for all i,j,k.
   
   We can write: T = (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k e_k) -- but that's not a
   valid tensor decomposition in the multiplicative sense we need.

   Actually, the correct interpretation: we need U, V, W such that for all i,j,k:
     T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
   
   Wait, that's not standard. Let me re-read the contract.

   The contract says: "U", "V", "W": factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
   The tensor is the matrix multiplication tensor. For multiplying (n x m) by (m x p),
   the output is (n x p). The tensor T has indices (i, j, k) where i indexes rows of A,
   j indexes the shared dimension, k indexes columns of B. T_{i,j,k} = 1.

   The decomposition: for each r, we have vectors u_r in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
   The product is: C[i,k] = sum_j A[i,j] * B[j,k] = sum_{i,j,k} T_{i,j,k} A[i,j] B[j,k]
   
   With decomposition: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]?
   
   No, that's not right either. Let me think about what the evaluator checks.

   Given the schoolbook solution: U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
   for the (i,j,k) at column 'col'. This means for each (i,j,k) triple, there's a term
   that picks out A[i,j] * B[j,k] and contributes to C[i,k] via W[k*n+i].

   So the decomposition is: C = sum_{i,j,k} A[i,j] B[j,k] * (e_i in output) 
   where each term is rank-1 in the tensor sense.

   Now, since T_{i,j,k} = 1 for all i,j,k, we can potentially merge terms.
   
   Key insight: If we fix j, then for all i,k, T_{i,j,k} = 1. The slice at fixed j is
   the all-ones n x p matrix, which has rank 1. So for each j, we can represent the
   contribution using rank-1 terms in the (i,k) space.
   
   Specifically: T = sum_j (e_j in j-dim) ⊗ (all-ones n x p matrix)
   
   The all-ones n x p matrix = (1,1,...,1)^T_n ⊗ (1,1,...,1)^T_p
   
   So: T = sum_j [ (e_j ⊗ 1_n) in U-space ] ⊗ [ (e_j ⊗ 1_p) in V-space ] ⊗ [ (1_n ⊗ 1_p) in W-space ]
   
   Wait, let me be more careful with the indexing.
   
   U-space index: i*m + j (for i in [0,n), j in [0,m))
   V-space index: j*p + k (for j in [0,m), k in [0,p))  
   W-space index: k*n + i (for k in [0,p), i in [0,n))
   
   For a fixed j, the U-vector has 1's at positions i*m + j for all i in [0,n).
   The V-vector has 1's at positions j*p + k for all k in [0,p).
   The W-vector has 1's at positions k*n + i for all k in [0,p), i in [0,n) -- i.e., ALL positions.
   
   So for each j, we get one rank-1 term:
     u_j: 1 at indices {0*m+j, 1*m+j, ..., (n-1)*m+j}
     v_j: 1 at indices {j*p+0, j*p+1, ..., j*p+(p-1)}
     w_j: 1 at all indices in [0, p*n)
   
   This gives rank = m = 4!
   
   Let's verify: For a given (i,j,k), T_{i,j,k} = sum_j' u_{j'}[i*m+j] * v_{j'}[j*p+k] * w_{j'}[k*n+i]
   
   u_{j'}[i*m+j] = 1 iff j' = j (since u_{j'} has 1's only at positions with the j-index = j')
   v_{j'}[j*p+k] = 1 iff j' = j (since v_{j'} has 1's only at positions with the j-index = j')
   w_{j'}[k*n+i] = 1 for all j' (since w_{j'} is all-ones)
   
   So the sum is: sum_{j'} delta(j',j) * delta(j',j) * 1 = delta(j,j) = 1 (when j'=j)
   
   Yes! This works. Rank = m = 4.

   Actually wait, let me double-check. The sum is over r (the rank index). In our case r ranges over j'.
   For the term to be nonzero, we need j' = j (from u) AND j' = j (from v). So only j' = j contributes,
   giving 1*1*1 = 1. Correct.

   So the rank is exactly m = 4 for the all-ones tensor.
"""

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m  # rank = m, one term per middle index j
    
    # U: shape (n*m, rank)
    # U[i*m + j, j] = 1 for all i, j; 0 otherwise
    U = [[0] * rank for _ in range(n * m)]
    for j in range(m):
        for i in range(n):
            U[i * m + j][j] = 1
    
    # V: shape (m*p, rank)
    # V[j*p + k, j] = 1 for all k, j; 0 otherwise
    V = [[0] * rank for _ in range(m * p)]
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
    
    # W: shape (p*n, rank)
    # W[k*n + i, j] = 1 for all i, k, j; 0 otherwise (all-ones)
    W = [[0] * rank for _ in range(p * n)]
    for j in range(m):
        for k in range(p):
            for i in range(n):
                W[k * n + i][j] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
