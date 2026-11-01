"""
Strategy: Direct Tensor Decomposition Search via Randomized Gradient Descent with Projection
Instead of relying on fixed algebraic constructions like Kronecker products or block decompositions,
this approach treats the tensor decomposition problem as a continuous optimization problem.

We seek rank r factors U, V, W such that the matrix multiplication tensor T_{2,4,6} can be
written as sum_{k=1}^r u_k v_k^T w_k^T (in the appropriate flattened form).
The tensor T is defined by T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k, and 0 otherwise.
Actually, the tensor is sparse. It has non-zero entries only where the indices match the 
multiplication structure. Specifically, for each (i,j,k), there is exactly one non-zero term 
in the tensor which is 1. Wait, the tensor is defined such that 
C_{k,i} = sum_j A_{i,j} B_{j,k}.
This corresponds to a tensor T where T_{(i,j),(j,k),(k,i)} = 1.
The total number of non-zero entries is n*m*p = 2*4*6 = 48.

We can formulate the problem as:
Minimize || T - sum_{k=1}^r u_k v_k^T w_k^T ||_F^2
where u_k, v_k, w_k are vectors of lengths n*m, m*p, p*n respectively.
Since the target tensor T has integer entries (0 or 1), and we want an exact decomposition,
we can use this optimization to find a candidate decomposition, and then round the entries 
to the nearest integer or half-integer to check for exactness.

However, standard gradient descent might get stuck in local minima. 
A more robust approach for small ranks is to use a randomized initialization and then 
perform a local search or use a solver that can handle the non-convexity.
Given the small dimensions (48, 24, 12), we can try ranks starting from a lower bound.
The lower bound for 2x4x6 is not known exactly, but it is at most 42 (from Strassen blocks).
We will try ranks from 30 to 42.
For each rank, we will perform several random restarts of gradient descent.
If we find a solution where the residual is close to zero, we will attempt to "snap" the 
entries to the nearest integer or half-integer and verify if it gives an exact decomposition.

To speed up the search, we can use the fact that the tensor is sparse.
We can represent the tensor as a list of non-zero entries.
Let NZ be the list of (index_u, index_v, index_w) for the 48 non-zero entries.
The residual is sum_{(iu, iv, iw) in NZ} (T[iu,iv,iw] - sum_k U[iu,k] V[iv,k] W[iw,k])^2
Since T[iu,iv,iw] = 1 for all NZ, this is sum_{(iu,iv,iw) in NZ} (1 - sum_k U[iu,k] V[iv,k] W[iw,k])^2.

We can use numpy for efficient computation.
The gradient with respect to U[iu,k] is -2 * sum_{(iu', iv, iw) in NZ, iu'=iu} (1 - S[iu,iv,iw]) * V[iv,k] * W[iw,k].
This can be computed efficiently using matrix operations.

Let's define:
S[iu, iv, iw] = sum_k U[iu,k] V[iv,k] W[iw,k].
This is a tensor product.
We can compute the residual and gradients using einsum.

Let U be (n*m, r), V be (m*p, r), W be (p*n, r).
The tensor S is of shape (n*m, m*p, p*n).
S = np.einsum('ik,jk,lk->ijl', U, V, W)
The residual is the sum of squares of (1 - S) over the non-zero entries of T.
Since T is zero elsewhere, and we only care about the non-zero entries for the residual 
(if we assume the decomposition is exact, it must be zero elsewhere too, but the gradient 
from the zero entries will push the solution towards zero there, which is good).
Actually, we should include all entries in the residual to ensure the decomposition is 
zero where T is zero.
So Residual = || T - S ||_F^2.
T is a (n*m, m*p, p*n) tensor with 1s at the 48 positions and 0s elsewhere.
We can create T as a dense array since the dimensions are small (8, 24, 12).
8*24*12 = 2304 elements. This is very small.

So we can just use dense arrays and numpy's einsum for the optimization.

Algorithm:
1. Define T.
2. For rank in [30, 31, ..., 42]:
   a. For trial in range(num_trials):
      i. Initialize U, V, W randomly with small values.
      ii. Perform gradient descent with adaptive learning rate.
      iii. If residual < threshold, snap to nearest integer/half-integer and verify.
      iv. If verified, return the solution.
3. If no solution found, fall back to the rank-42 Strassen solution.

This is a fundamentally different strategy: it's a direct numerical search for the 
decomposition, rather than a constructive algebraic method.
"""

import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Create the target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Fallback: Rank 42 solution (6 blocks of 2x2x2 Strassen)
    def get_rank_42_solution():
        U_s = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ])
        V_s = np.array([
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 0, -1, 0, 1, 0, 1]
        ])
        W_s = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ])
        
        rank = 42
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        def add_strassen_block(U, V, W, A_i, A_j, B_j, B_k, C_k, C_i, offset):
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        global_i = A_i[li]
                        global_j = A_j[lj]
                        U[global_i * m + global_j, offset + r] += U_s[li*2 + lj, r]
                for lj in range(2):
                    for lk in range(2):
                        global_j = B_j[lj]
                        global_k = B_k[lk]
                        V[global_j * p + global_k, offset + r] += V_s[lj*2 + lk, r]
                for lk in range(2):
                    for li in range(2):
                        global_k = C_k[lk]
                        global_i = C_i[li]
                        W[global_k * n + global_i, offset + r] += W_s[lk*2 + li, r]
        
        add_strassen_block(U, V, W, [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0)
        add_strassen_block(U, V, W, [0,1], [2,3], [2,3], [0,1], [0,1], [0,1], 7)
        add_strassen_block(U, V, W, [0,1], [0,1], [0,1], [2,3], [2,3], [0,1], 14)
        add_strassen_block(U, V, W, [0,1], [2,3], [2,3], [2,3], [2,3], [0,1], 21)
        add_strassen_block(U, V, W, [0,1], [0,1], [0,1], [4,5], [4,5], [0,1], 28)
        add_strassen_block(U, V, W, [0,1], [2,3], [2,3], [4,5], [4,5], [0,1], 35)
        
        return U.tolist(), V.tolist(), W.tolist()

    # Optimization function
    def optimize(rank, num_trials, max_iters, lr_init):
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            lr = lr_init
            prev_loss = np.inf
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                    
                # Compute S = U V^T W^T (tensor product)
                S = np.einsum('ik,jk,lk->ijl', U, V, W)
                residual = T - S
                loss = np.sum(residual**2)
                
                # Check convergence
                if loss < 1e-6:
                    # Snap to nearest integer or half-integer
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify exactness
                    S_snap = np.einsum('ik,jk,lk->ijl', U_snap, V_snap, W_snap)
                    if np.allclose(S_snap, T, atol=1e-9):
                        return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                    break
                
                # Compute gradients
                # dL/dU[i,k] = -2 * sum_{j,l} residual[i,j,l] * V[j,k] * W[l,k]
                grad_U = -2 * np.einsum('ijl,jk,lk->ik', residual, V, W)
                grad_V = -2 * np.einsum('ijl,ik,lk->jk', residual, U, W)
                grad_W = -2 * np.einsum('ijl,ik,jk->lk', residual, U, V)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Adaptive learning rate
                if loss < prev_loss:
                    lr *= 1.05
                else:
                    lr *= 0.9
                    # Undo update if loss increased significantly
                    U += 2 * lr * grad_U
                    V += 2 * lr * grad_V
                    W += 2 * lr * grad_W
                
                prev_loss = loss

    # Try to find a solution with rank < 42
    for rank in range(30, 42):
        if time.time() > deadline - 20:
            break
        result = optimize(rank, num_trials=5, max_iters=200, lr_init=0.01)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to rank 42
    U, V, W = get_rank_42_solution()
    
    # Verify
    def check(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for rr in range(r):
                        s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not check(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
    return {"U": U, "V": V, "W": W}
