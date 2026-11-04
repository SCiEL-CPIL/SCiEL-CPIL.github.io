"""
Matrix multiplication tensor decomposition for 3x3 * 3x5.

We implement a fast, correct schoolbook (rank-45) decomposition and then
attempt to find a lower-rank exact decomposition via a structured search
that exploits the algebraic structure of matrix multiplication.

For A (3x3) * B (3x5) = C (3x5):
  C[i,k] = sum_j A[i,j] * B[j,k]

The tensor T has entries T[a,b,c] = 1 where:
  a = i*m + j  (i in [0,3), j in [0,3))
  b = j*p + k  (j in [0,3), k in [0,5))
  c = k*n + i  (k in [0,5), i in [0,3))  -- trace convention W[k*n+i]

Schoolbook: rank = 3*3*5 = 45.

We try to find a lower rank by using a structured approach:
- The tensor is a "matrix multiplication tensor" with specific structure.
- We can try to exploit the fact that for each fixed j, the sub-tensor
  over (i,k) is a rank-1 tensor (outer product of e_i and e_k).
  
A key observation: For each column j of A and row j of B, we have
  C[:,k] contribution from j: A[:,j] * B[j,k]
  
This means the tensor can be written as sum over j of (A[:,j] outer B[j,:]).
In tensor form, for each j, the contribution to T is:
  For each (i,k): T[i*m+j, j*p+k, k*n+i] += 1

This is already the schoolbook. To get lower rank, we need to combine
multiple (i,j,k) triples into single rank-1 terms.

For 3x3 matrix multiplication, Strassen's algorithm gives rank 7 instead of 9.
For 3x3 * 3x5, we can potentially use a similar approach.

Let me think about this more carefully. The tensor is:
  T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}  (in the appropriate spaces)

For the 3x3 case (p=3), Strassen combines the 9 terms into 7.
For 3x3 * 3x5, we have 45 terms. We can potentially use the Strassen
decomposition on the 3x3 part and extend it.

Actually, let me try a different approach. Let's use the fact that:
  C = A * B

where A is 3x3 and B is 3x5. We can write:
  C[:, k] = A * B[:, k]  for each k

If we use Strassen for the 3x3 matrix-vector product (which is just 3x3 * 3x1),
the rank for 3x3 * 3x1 is... well, it's the same as 3x3 matrix multiplication
restricted to a single column, so the tensor is T[a,b,c] where b = j*1+k = j,
c = k*3+i = i (since p=1, k=0 only). Wait, this doesn't quite work.

Let me try a more direct approach: use a randomized search with verification,
but with a better initialization and more steps. The key insight is that
the solution space is a variety, and we can try to find points on it.

Actually, for this specific problem, let me try to construct a decomposition
using the following idea:

For each of the 5 columns of B, we need to compute A * B[:,k].
If we can find a rank-r decomposition for 3x3 * 3x1 (matrix-vector),
then we can potentially reuse it for all 5 columns.

For 3x3 * 3x1:
  C[i] = sum_j A[i,j] * B[j]
  T[a,b,c] = 1 where a=i*3+j, b=j (since p=1, b=j*1+0=j), c=0*3+i=i

So T has size (9) x (3) x (3), with T[i*3+j, j, i] = 1.

This is a 9x3x3 tensor. The schoolbook rank is 9 (one term per (i,j)).
Can we do better? For 3x3 * 3x1, the minimum rank is at least 6 (by
some lower bounds) and at most 9 (schoolbook). Actually, I think the
rank might be 6 or 7.

Let me try to find a rank-6 decomposition for 3x3 * 3x1:
  We need U (9x6), V (3x6), W (3x6) such that
  sum_k U[i*3+j,k] * V[j,k] * W[i,k] = 1 if valid, 0 otherwise.

This is a specific tensor. Let me try to solve it.

Actually, this is getting complex. Let me just implement a robust schoolbook
solution and a quick randomized search, and return the best found.

Given the time constraints, I'll focus on:
1. A correct schoolbook (rank 45)
2. A quick attempt at rank 30 using structured random search
3. Return the best valid decomposition found
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe

    # Build the target tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook decomposition
    def schoolbook():
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, r

    best_U, best_V, best_W, best_rank = schoolbook()

    def verify(U, V, W):
        P = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.max(np.abs(P - T))

    # Try to find lower rank decompositions
    # Use a structured approach: try ranks from 35 down to 25
    for target_rank in [35, 32, 30]:
        if time.time() > deadline - 15:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline - 15:
                break
            
            # Initialize with random values
            np.random.seed(42 + attempt)
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Gradient descent with adaptive learning rate
            lr = 0.1
            for step in range(500):
                if time.time() > deadline - 15:
                    break
                
                P = np.einsum('ak,bk,ck->abc', U, V, W)
                diff = P - T
                err = np.max(np.abs(diff))
                
                if err < 1e-8:
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W
                    found = True
                    break
                
                # Compute gradients
                grad_U = np.einsum('abc,bk,ck->ak', diff, V, W)
                grad_V = np.einsum('abc,ak,ck->bk', diff, U, W)
                grad_W = np.einsum('abc,ak,bk->ck', diff, U, V)
                
                # Normalize gradients
                gU_norm = np.linalg.norm(grad_U)
                gV_norm = np.linalg.norm(grad_V)
                gW_norm = np.linalg.norm(grad_W)
                
                if gU_norm > 1e-10:
                    grad_U /= gU_norm
                if gV_norm > 1e-10:
                    grad_V /= gV_norm
                if gW_norm > 1e-10:
                    grad_W /= gW_norm
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Adaptive learning rate
                if step % 50 == 0 and step > 0:
                    lr *= 0.8
            
            if found:
                break
        
        if found:
            break

    # Final verification
    if verify(best_U, best_V, best_W) > 1e-6:
        # Fall back to schoolbook if something went wrong
        best_U, best_V, best_W, best_rank = schoolbook()

    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
