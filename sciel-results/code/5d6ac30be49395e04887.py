"""Different strategy: use bilinear algebra (Strassen-style / rank-reduction via linear algebra)
to find a lower-rank decomposition of the 4x5x5 matrix multiplication tensor.

The key insight: the matrix multiplication tensor has structure. We can view it as
a 3-way tensor T[i,j,k] where the bilinear map is C[i,k] = sum_j A[i,j] B[j,k].

A different approach: instead of brute-force rank-1 search, we exploit the fact that
the tensor for (n,m,p) matrix multiplication can be decomposed using the structure
that C = A*B. We can try to find a rank-r decomposition by solving a system of
polynomial equations, but more practically, we can use the known bounds and try
specific structured decompositions.

For 4x5x5, the schoolbook rank is 100. The best known upper bounds from literature
suggest it could be lower. Let's try a different approach:

1. Represent the tensor as a matrix (matricization) and find its rank.
2. Use the fact that if we can find a rank-r decomposition, we need:
   U (20 x r), V (25 x r), W (20 x r) such that for all i,k:
   sum_r U[i*5+j, r] * V[j*5+k, r] = delta_{i,k} ... wait, that's not right.

Actually, let me reconsider the tensor structure. For C = A*B where A is 4x5, B is 5x5:
C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]

The tensor T is in R^{20} ⊗ R^{25} ⊗ R^{20} (flattened).
T is defined by: for each (i,j,k), the term A[i,j]*B[j,k] contributes to C[i,k].

In the factorization format:
- U has shape (n*m, rank) = (20, rank)
- V has shape (m*p, rank) = (25, rank)  
- W has shape (p*n, rank) = (20, rank)

And the constraint is: for all i in [0,4), k in [0,5):
  sum_{r=0}^{rank-1} sum_{j=0}^{4} U[i*5+j, r] * V[j*5+k, r] * W[k*4+i, r] ... 

Wait, that's not right either. Let me think more carefully.

The bilinear form is: C[i,k] = sum_j A[i,j] B[j,k]

In tensor decomposition form, we want:
C[i,k] = sum_r U[i*5 + ?, r] ... 

Actually, the standard way to express this: the tensor T has entries T[i,j,k] = 1 for all i,j,k (since each product A[i,j]*B[j,k] appears in exactly one C entry, namely C[i,k]). But that's the all-ones tensor which has rank 1, which is wrong.

Let me reconsider. The correct tensor for matrix multiplication:
T[i,j,k] where i indexes (row of A, col of C), j indexes (col of A, row of B), k indexes (col of B, col of C).
The constraint is that T[i,j,k] = 1 if and only if the (i,j) entry of A and (j,k) entry of B multiply to contribute to the (i,k) entry of C.

Since for each (i,j,k), the product A[i,j]*B[j,k] contributes to C[i,k], we have T[i,j,k] = 1 for all valid (i,j,k). But this is a 4x5x5 tensor of all ones, which has rank 1. That can't be right for matrix multiplication.

I think I'm confusing the tensor with the structure. Let me look at this differently.

The matrix multiplication map is: (A, B) -> C where C = A*B.
This is a bilinear map from R^{4x5} x R^{5x5} -> R^{4x5}.

The tensor associated with this bilinear map is T in R^{20} ⊗ R^{25} ⊗ R^{20} such that:
C[i,k] = sum_{a=0}^{19} sum_{b=0}^{24} T[a,b,c] * A_flat[a] * B_flat[b] where c indexes the output.

Wait, I think the issue is that the tensor T is defined such that:
C_flat[c] = sum_{a,b} T[a,b,c] * A_flat[a] * B_flat[b]

where A_flat has 20 elements (4*5), B_flat has 25 elements (5*5), C_flat has 20 elements (4*5).

For C[i,k] = sum_j A[i,j] B[j,k], we have:
C_flat[i*5+k] = sum_{j=0}^{4} A_flat[i*5+j] * B_flat[j*5+k]

So T[a,b,c] = 1 if a = i*5+j, b = j*5+k, c = i*5+k for some i,j,k.

This is a sparse tensor. The rank of this tensor is what we want to minimize.

Now, a different strategy: use the fact that this tensor can be viewed as a matrix (matricization) and use SVD or other linear algebra techniques to find a low-rank approximation, then round to get an exact decomposition.

But for exact decomposition with integer/half-integer entries, we need a different approach.

Let me try a constructive approach: for small dimensions, we can try to find a rank-r decomposition by solving a system of equations. For r < 100, we need to find U, V, W such that:
sum_r U[i*5+j, r] * V[j*5+k, r] = delta_{i,k} ... no, that's not right.

Actually, the constraint is:
For all i in [0,4), k in [0,5):
  C[i,k] = sum_{r=0}^{r-1} (sum_{j=0}^{4} U[i*5+j, r] * V[j*5+k, r]) * W[k*4+i, r]

Wait, that's not the standard form. Let me look at the interface again.

The interface says: U (n*m, rank), V (m*p, rank), W (p*n, rank).
And the output factor follows the official trace convention W[k*n+i, r].

So the bilinear form is:
C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

Wait, that doesn't make sense dimensionally. Let me re-read.

"Factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"

So U is 20 x rank, V is 25 x rank, W is 20 x rank.

The bilinear form for matrix multiplication should be:
C[i,k] = sum_{r} U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] ... but j is summed over.

Actually, I think the correct form is:
C[i,k] = sum_{r} (sum_{j} U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]

But that's not standard. Let me think about this differently.

In a tensor decomposition, T = sum_r u_r ⊗ v_r ⊗ w_r, where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.

The bilinear map is: (A, B) -> C where C[i,k] = sum_{j} A[i,j] B[j,k].

In terms of the tensor: C_flat[c] = sum_{a,b} T[a,b,c] A_flat[a] B_flat[b]

where T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k.

Wait, c should index the output C which is n x p, so c = i*p + k or c = k*n + i depending on the convention. The problem says "W[k*n+i, r]" so c = k*n + i.

So T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i.

For n=4, m=5, p=5:
a = i*5+j for i in [0,4), j in [0,5)
b = j*5+k for j in [0,5), k in [0,5)
c = k*4+i for k in [0,5), i in [0,4)

The tensor T is 20 x 25 x 20.

A rank-r decomposition means: T = sum_{r=0}^{R-1} u_r ⊗ v_r ⊗ w_r
where u_r is 20-dim, v_r is 25-dim, w_r is 20-dim.

Equivalently, for all (a,b,c):
T[a,b,c] = sum_{r} U[a,r] * V[b,r] * W[c,r]

This is a system of 20*25*20 = 10000 equations in 20R + 25R + 20R = 65R unknowns.

For R < 100, this is an underdetermined system (65R < 10000 for R < 154), so solutions may exist.

A different strategy: use random initialization and gradient descent to minimize the reconstruction error, then round to the nearest integer/half-integer and verify.

Let me implement this:
1. Initialize U, V, W randomly.
2. Use gradient descent to minimize ||T - sum_r u_r ⊗ v_r ⊗ w_r||_F^2.
3. Round to nearest 0.5 and verify exactness.
4. If not exact, try again with different seeds or increase R.

This is a fundamentally different approach from the schoolbook decomposition.

"""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin

    # Build the target tensor T of shape (n*m, m*p, p*n) = (20, 25, 20)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank_sb = n * m * p
        U = np.zeros((n*m, rank_sb))
        V = np.zeros((m*p, rank_sb))
        W = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank_sb

    # Try to find a lower-rank decomposition using gradient descent
    best_result = None
    best_rank = n * m * p

    U_sb, V_sb, W_sb, _ = schoolbook()
    best_result = (U_sb, V_sb, W_sb, best_rank)

    # Try ranks from 80 down to 60 (or as far as time allows)
    for rank in range(80, 59, -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        found = False
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + trial * 1000 + rank)
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Gradient descent
            lr = 0.01
            for step in range(5000):
                if time.time() > deadline - 5:
                    break
                
                # Compute current decomposition
                # T_approx[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # Efficient computation: for each r, outer product
                T_approx = np.zeros_like(T)
                for r in range(rank):
                    T_approx += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                
                error = np.sum((T - T_approx) ** 2)
                if error < 1e-10:
                    # Round to nearest 0.5
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify
                    T_check = np.zeros_like(T)
                    for r in range(rank):
                        T_check += np.einsum('a,b,c->abc', U_r[:, r], V_r[:, r], W_r[:, r])
                    if np.allclose(T_check, T, atol=1e-9):
                        best_result = (U_r, V_r, W_r, rank)
                        best_rank = rank
                        found = True
                        break
                
                # Compute gradients
                # dT/du_a = sum_{b,c} (T[a,b,c] - T_approx[a,b,c]) * V[b,r] * W[c,r]
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                for r in range(rank):
                    # dU[:,r] = (T[:, :, :] - T_approx[:, :, :]) contracted with V[:,r] and W[:,r]
                    dU[:, r] = np.einsum('abc,b,c->a', T - T_approx, V[:, r], W[:, r])
                    dV[:, r] = np.einsum('abc,a,c->b', T - T_approx, U[:, r], W[:, r])
                    dW[:, r] = np.einsum('abc,a,b->c', T - T_approx, U[:, r], V[:, r])
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            if found:
                break
        
        if found:
            break
    
    U, V, W, rank = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
