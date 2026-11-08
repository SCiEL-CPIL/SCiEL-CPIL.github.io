"""
Rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.

Uses a block decomposition approach:
- Split the 3x3 matrix A into blocks and apply Strassen-like ideas
- The key insight: we can decompose the multiplication into smaller sub-problems
  and use known low-rank decompositions for each sub-problem

For 3x3 * 3x7, we can view this as computing 3 rows of C, each being a 1x3 * 3x7 product.
But more effectively, we can use a block structure where we split columns of B.

Strategy:
1. Schoolbook: rank 63
2. Try to find structured decompositions with lower rank
3. Use the fact that 3x3 matrix multiplication can be done with rank 23 (or lower constructions)
   and extend this to the 3x7 case

Actually, let's think about it differently:
C = A * B where A is 3x3 and B is 3x7.
We can write B as [B1, B2, B3, B4, B5, B6, B7] where each Bj is 3x1.
Then C = [A*B1, A*B2, ..., A*B7].

Each A*Bj is a 3x1 vector = A times a 3x1 vector.
This is a 3x3 * 3x1 multiplication, which has rank 3 (trivially) or potentially less.

But the A is shared across all 7 products. So we need a joint decomposition.

Let me try a different approach: use the tensor decomposition framework directly.
We need to find U (9 x r), V (21 x r), W (21 x r) such that for all i, j, k:
sum_t U[i*3+j, t] * V[j*7+k, t] * W[k*3+i, t] = 1 if the indices match correctly, else 0.

Actually, the correct condition is:
For the bilinear map (A, B) -> C where C[i,k] = sum_j A[i,j]*B[j,k],
the tensor decomposition requires:
sum_t U[a, t] * V[b, t] * W[c, t] = T[a,b,c]
where a = i*3+j (index into A), b = j*7+k (index into B), c = k*3+i (index into C),
and T[a,b,c] = 1 if the indices are consistent (i.e., the j in a matches the j in b, 
and i in a matches i in c, and k in b matches k in c), else 0.

Wait, let me re-examine. The bilinear form is:
C[i,k] = sum_j A[i,j] * B[j,k]

In tensor notation, if we flatten A to a vector of length 9 (index i*3+j),
B to a vector of length 21 (index j*7+k), and C to a vector of length 21 (index k*3+i),
then the multiplication tensor T has:
T[(i*3+j), (j'*7+k), (k'*3+i')] = 1 if i=i' and j=j' and k=k', else 0.

No wait, that's not right. Let me be very careful.

The bilinear map is: given A (3x3) and B (3x7), compute C (3x7) where C[i,k] = sum_j A[i,j] B[j,k].

In the tensor decomposition framework, we want to write:
C = sum_t (U_t · A) * (V_t · B) * W_t
where U_t is a 9-dimensional vector, V_t is a 21-dimensional vector, W_t is a 21-dimensional vector,
and (U_t · A) means sum_{i,j} U_t[i*3+j] * A[i,j], etc.

But that gives a scalar, not a matrix. So the correct formulation is:
The output C[i,k] is a linear combination of products of linear forms in A and B.

Specifically: C[i,k] = sum_t alpha_{i,k,t} * (sum_{i',j'} U_t[i'*3+j'] * A[i',j']) * (sum_{j',k'} V_t[j'*7+k'] * B[j',k'])

For this to equal sum_j A[i,j] B[j,k] for all A, B, we need:
sum_t alpha_{i,k,t} * U_t[i'*3+j'] * V_t[j'*7+k'] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}

Wait, that's a 3-index equation. Let me use the standard matrix multiplication tensor formulation.

The multiplication tensor for (n,m,p) has entries:
M[(i,j), (j',k), (i',k')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}

A rank-r decomposition is: M = sum_{t=1}^r u_t ⊗ v_t ⊗ w_t
where u_t is in R^{n*m}, v_t is in R^{m*p}, w_t is in R^{p*n}.

So: M[(i,j), (j',k), (i',k')] = sum_t u_t[i*m+j] * v_t[j'*p+k] * w_t[k'*n+i']

This must equal delta_{i,i'} * delta_{j,j'} * delta_{k,k'}.

For n=3, m=3, p=7:
- u_t has 9 entries (indexed by i*3+j)
- v_t has 21 entries (indexed by j'*7+k)  
- w_t has 21 entries (indexed by k'*3+i')

And we need: sum_t u_t[i*3+j] * v_t[j'*7+k] * w_t[k'*3+i'] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}

The schoolbook decomposition has rank 63: for each (i,j,k), set u[i*3+j]=1, v[j*7+k]=1, w[k*3+i]=1.

Can we do better? For 3x3 * 3x3, the best known rank is 23. For 3x3 * 3x7, we might be able to do better than 63.

One approach: split the 7 columns of B into groups and use block decomposition.
For example, split B into [B1 | B2] where B1 is 3x3 and B2 is 3x4.
Then C = [A*B1 | A*B2].

The A*B1 part is 3x3 * 3x3, which can be done with rank 23 (using known constructions).
The A*B2 part is 3x3 * 3x4, which can be done with rank... let's think.

Actually, this splitting doesn't directly help because the A is shared.

Let me try a more direct approach: use a structured ansatz based on the algebraic structure.

For 3x3 matrix multiplication, one known rank-23 decomposition uses the following idea:
Write A and B as block matrices and use Strassen-like products.

But for 3x3 * 3x7, let me try a simpler approach first:
Can we achieve rank 42 or lower with a simple structure?

Let me try: for each i (row of A/C), use 7 terms (one per column k of B/C), but share the j-index.
This would give rank 3*7 = 21, but that's too optimistic.

Actually, let's just implement a verification function and try a few structured ansatzes.
"""

import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def verify_decomposition(U, V, W, n=3, m=3, p=7, tol=1e-9):
    """
    Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    
    Condition: sum_t U[i*m+j, t] * V[j'*p+k, t] * W[k'*n+i', t] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    for all i, j, j', k, k', i'.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            computed = 0.0
                            for t in range(rank):
                                computed += U[i*m+j, t] * V[jp*p+k, t] * W[kp*n+ip, t]
                            if abs(computed - expected) > tol:
                                return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try structured decompositions with lower rank
    # Strategy: use block decomposition of the multiplication
    
    # Approach 1: Split columns of B into groups
    # B = [B1 | B2] where B1 is 3x3, B2 is 3x4
    # C = [A*B1 | A*B2]
    # We can use a rank-23 decomposition for 3x3*3x3 and extend it
    
    # For now, let's try to find any decomposition with rank < 63
    # by solving a linear system for a structured ansatz
    
    # Try rank 45, 42, 40, 36, 35, 30, 28, 25, 24, 23, 22, 21
    target_ranks = [45, 42, 40, 36, 35, 30, 28, 25, 24, 23, 22, 21]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        # Try a structured ansatz: assume the decomposition has a block structure
        # where for each i, the U entries are sparse
        
        # Simple approach: try to find a rank-r decomposition by solving for U, V, W
        # given a parameterization
        
        # For now, just try a few random structured ansatzes
        for trial in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(12345 + trial + target_rank)
            
            # Random initialization
            U_r = np.random.randn(n * m, target_rank)
            V_r = np.random.randn(m * p, target_rank)
            W_r = np.random.randn(p * n, target_rank)
            
            # Normalize
            for t in range(target_rank):
                norm = np.sqrt(np.sum(U_r[:, t]**2) * np.sum(V_r[:, t]**2) * np.sum(W_r[:, t]**2))
                if norm > 1e-10:
                    U_r[:, t] /= norm
                    V_r[:, t] /= norm
                    W_r[:, t] /= norm
            
            # Verify
            if verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                best_rank = target_rank
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                break
    
    return best_result
