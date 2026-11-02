"""
Matrix multiplication 4x4 by 4x5: rank-minimal exact tensor decomposition.

Approach:
1. Try known low-rank decompositions (e.g., from literature or structured patterns).
2. Use a border-rank / gradient-descent heuristic to find candidate low-rank decompositions.
3. Snap to half-integers via rational reconstruction.
4. Verify exactness.
5. Return the best valid decomposition found.

For 4x4 * 4x5, the schoolbook rank is 4*4*5 = 80.
Theoretical lower bound is max(4*4, 4*5, 4*5) = 20, but actual min rank is unknown.
Known: for n x m * m x p, rank <= n*m*p. Some small cases have known optimal ranks.

We'll try:
- Start with a structured initialization based on Strassen-like or other known algorithms.
- Perform gradient descent on the tensor residual.
- Snap to half-integers.
- Verify.

Given the 120s time limit, we need to be efficient.

Let's think about what's achievable:
- The task says "integer/half-integer entries"
- Score is the rank (lower is better)
- Seed program uses rank 80 (schoolbook)

We need to find a decomposition with rank < 80 that is exact.

For 4x4 * 4x5:
- One can use block decomposition: split the 4x4 into 2x2 blocks, etc.
- But we need exact half-integer entries.

Let me try a different approach: use known results or try to find a decomposition with rank around 40-60.

Actually, let me think about this more carefully. The matrix multiplication tensor for A(4x4) * B(4x5) = C(4x5) has:
- Input: A has 16 entries, B has 20 entries, Output C has 20 entries
- Tensor is in R^{16} ⊗ R^{20} ⊗ R^{20}

The rank of this tensor is what we're minimizing.

Known bounds:
- Upper bound: 16*20 = 320? No, schoolbook is n*m*p = 4*4*5 = 80.
- Lower bound: max(16, 20, 20) = 20 (but this is a very weak bound)

For small matrix multiplication, the exact minimal rank is often unknown. But we can try to find good decompositions.

Strategy:
1. Use a random initialization and gradient descent to minimize the Frobenius norm of the residual.
2. After convergence, snap to half-integers.
3. Verify exactness.
4. If successful, return the decomposition.

But gradient descent on tensor rank minimization is non-convex and may not converge to a good solution.

Alternative: Try to use known structured decompositions.

Let me try a block-based approach:
- Split A into 2x2 blocks of 2x2 matrices
- Split B into 2x2 blocks of 2x2 matrices (but B is 4x5, so this is tricky)

Actually, B is 4x5, so it doesn't split evenly into 2x2 blocks.

Let me try another approach: use the fact that we can express the multiplication using a smaller number of multiplications by exploiting linearity.

For now, let me implement a practical approach:
1. Try several random initializations
2. Use gradient descent to reduce the rank
3. Snap to half-integers
4. Verify

Given time constraints, I'll implement a simplified version.
"""

import numpy as np
from numpy.linalg import norm

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Build the target tensor
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # Actually, the multiplication tensor is defined as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[a, b, c] where a indexes A, b indexes B, c indexes C
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k) for some i,j,k, else 0
    
    # Let's build the tensor explicitly
    # A index: i*m + j, i in [0,n), j in [0,m)
    # B index: j*p + k, j in [0,m), k in [0,p)
    # C index: k*n + i, k in [0,p), i in [0,n)  (following W[k*n+i, r] convention)
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try to find a low-rank decomposition
    # Start with a reasonable rank and try to reduce
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    
    # Try various ranks
    for target_rank in [40, 45, 50, 55, 60, 65, 70, 75, 80]:
        # Initialize with random values
        np.random.seed(target_rank)
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Gradient descent on ||T - sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]||_F^2
        # The residual is R = T - sum_r U[:,r] V[:,r]^T W[:,r]^T (as a tensor)
        
        lr = 0.001
        for iteration in range(2000):
            # Compute residual
            R = T.copy()
            for r in range(target_rank):
                R -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            # Check if we're close to zero
            res_norm = norm(R)
            if res_norm < 1e-6:
                break
            
            # Compute gradients
            # d/dU[:,r] = sum over b,c: R[a,b,c] * V[b,r] * W[c,r]
            # d/dV[:,r] = sum over a,c: R[a,b,c] * U[a,r] * W[c,r]
            # d/dW[:,r] = sum over a,b: R[a,b,c] * U[a,r] * V[b,r]
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for r in range(target_rank):
                dU[:, r] = np.einsum('abc,bc->a', R, V[:, r], W[:, r])
                dV[:, r] = np.einsum('abc,ac->b', R, U[:, r], W[:, r])
                dW[:, r] = np.einsum('abc,ab->c', R, U[:, r], V[:, r])
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Snap to half-integers
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        
        # Verify
        R_snapped = T.copy()
        for r in range(target_rank):
            R_snapped -= np.einsum('a,b,c->abc', U_snapped[:, r], V_snapped[:, r], W_snapped[:, r])
        
        if norm(R_snapped) < 1e-9:
            if target_rank < best_rank:
                best_rank = target_rank
                best_U, best_V, best_W = U_snapped, V_snapped, W_snapped
            break
    
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
