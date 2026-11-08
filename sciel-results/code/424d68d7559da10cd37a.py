"""
Rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.

We use the standard schoolbook decomposition as a baseline (rank 48),
but try to find a lower-rank decomposition using known algebraic
identities or structured decompositions.

For matrix multiplication, the tensor T_{i,j,k} = e_i * e_j * e_k corresponds
to the (i,k)-entry of the product being the sum over j of A_{i,j} * B_{j,k}.

The tensor is of format (n, m, p) = (3, 4, 4), and we seek rank-r factors
U (n*m x r), V (m*p x r), W (p*n x r) such that:
    T_{(i,j),(j,k),(k,i)} = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}

Note: The W factor follows the official trace convention W[k*n + i, r].

Known lower bounds and results:
- Schoolbook: rank = n*m*p = 48
- For square n x n, Strassen gives rank ~ n^2.807
- For 3x3 * 3x3, best known is 23 (Landsberg)
- For 3x4 * 4x4, we can try to use block structure or known decompositions.

Let's try a direct approach: use the schoolbook but look for linear dependencies
among the 48 basis tensors to reduce the rank. However, for generic matrix
multiplication, the schoolbook decomposition is often near-optimal for small
sizes, but there may be savings.

Actually, let's think about this differently. The matrix multiplication tensor
T_{ijk} = 1 for all i,j,k (in the sense that each (i,j,k) triple contributes
to the product). Wait, no: the tensor for matrix multiplication C = A*B where
C_{ik} = sum_j A_{ij} B_{jk} has the property that the (i,j,k) entry of the
tensor is 1 (it's a "complete" tensor in the sense that all entries are 1).

So T is a 3x4x4 tensor with all entries equal to 1. We want to write this as
a sum of r rank-1 tensors: T_{ijk} = sum_{c=1}^r U_{ij,c} V_{jk,c} W_{ki,c}.

This is equivalent to finding the tensor rank of the all-ones tensor of shape
(3,4,4).

The rank of the all-ones tensor of shape (a,b,c) is at most a*b*c (schoolbook),
but can be lower. For example, the all-ones 2x2x2 tensor has rank 7 (Strassen),
not 8.

For our case, let's try to find a decomposition. One approach: use the fact that
the all-ones tensor can be decomposed using the "sum of products of sums" structure.

Let's try a constructive approach with random initialization and gradient descent,
or use known algebraic decompositions.

Actually, for the all-ones tensor T_{ijk} = 1, we can write:
1 = (1/2) * [ (x_i + x_j + x_k)^2 - x_i^2 - x_j^2 - x_k^2 - 2*x_i*x_j - 2*x_i*x_k - 2*x_j*x_k ] ... this isn't quite right.

Let me think more carefully. We want U_{ij,c} V_{jk,c} W_{ki,c} to sum to 1 for all i,j,k.

One simple decomposition: use the identity 1 = sum over some basis. But we need
the rank-1 structure with the specific index patterns.

Let's try a different approach: use the fact that for the all-ones tensor, we can
use a decomposition based on the "tensor rank of the all-ones matrix" generalized
to 3D.

Actually, let's just implement a numerical search. We'll initialize U, V, W randomly
and use gradient descent to minimize ||T - sum_c U_c V_c W_c||_F^2, then try to
drive the residual to zero and reduce the rank.

But for exactness, we need integer or half-integer entries. So we'll need to round
and verify.

Given the time constraint, let's try a few strategies:
1. Schoolbook (rank 48) - always works
2. Try to find a lower-rank decomposition via optimization
3. Use known algebraic identities if available

Let's implement a robust solution that tries optimization with a time limit,
falling back to schoolbook if needed.
"""

import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120 second limit
    
    # Build the target tensor T with shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # But actually, the tensor is indexed such that:
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # The constraint is that j is shared between a and b, and k,i are shared between b and c.
    
    # Let's build the target tensor properly.
    # The matrix multiplication tensor has entries T_{(i,j),(j,k),(k,i)} = 1
    # For our flattened indices:
    # U has shape (n*m, rank), indexed by (i, j) -> i*m + j
    # V has shape (m*p, rank), indexed by (j, k) -> j*p + k  
    # W has shape (p*n, rank), indexed by (k, i) -> k*n + i
    
    # The tensor T is a 3D array where T[i*m+j, j*p+k, k*n+i] = 1
    # But we only have entries where the indices are consistent (j matches, k and i match).
    
    # Actually, the way to think about it: the tensor T is defined on the full
    # (n*m) x (m*p) x (p*n) space, but most entries are 0. The non-zero entries
    # are T[i*m+j, j*p+k, k*n+i] = 1 for i in range(n), j in range(m), k in range(p).
    
    # Let's build this tensor.
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as fallback
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
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Strategy: for each candidate rank r, try random initialization + optimization
    # to see if we can achieve zero residual.
    
    def try_rank(r, max_iter=5000, lr=0.01, n_restarts=3):
        """Try to find a rank-r decomposition using gradient descent."""
        nonlocal best_U, best_V, best_W, best_rank
        if r >= best_rank:
            return False
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                return False
            
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.zeros((p * n, r))
            W[:, 0] = 1  # Initialize W with a non-zero column
            
            # Use Adam-like optimizer
            mU = np.zeros_like(U); vU = np.zeros_like(U)
            mV = np.zeros_like(V); vV = np.zeros_like(V)
            mW = np.zeros_like(W); vW = np.zeros_like(W)
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            t = 1
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    return False
                
                # Compute residual: R[a,b,c] = T[a,b,c] - sum_c U[a,c]V[b,c]W[c_idx,c]
                # where c_idx is the third index.
                # This is expensive, so let's compute it efficiently.
                
                # R = T - np.einsum('ac,bc,dc->abd', U, V, W)
                R = T - np.einsum('ac,bc,dc->abd', U, V, W)
                loss = np.sum(R ** 2)
                
                if loss < 1e-10:
                    # Found a solution!
                    # Round to nearest 0.5 and verify
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    R_check = T - np.einsum('ac,bc,dc->abd', U_r, V_r, W_r)
                    if np.max(np.abs(R_check)) < 1e-6:
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = r
                        return True
                    # If rounding doesn't work, try without rounding
                    R_check = T - np.einsum('ac,bc,dc->abd', U, V, W)
                    if np.max(np.abs(R_check)) < 1e-6:
                        best_U, best_V, best_W = U, V, W
                        best_rank = r
                        return True
                
                # Compute gradients
                # dL/dU[a,c] = -2 * sum_{b,d} R[a,b,d] * V[b,c] * W[d,c]
                # dL/dV[b,c] = -2 * sum_{a,d} R[a,b,d] * U[a,c] * W[d,c]
                # dL/dW[d,c] = -2 * sum_{a,b} R[a,b,d] * U[a,c] * V[b,c]
                
                # Using einsum for efficiency
                dU = -2 * np.einsum('abd,bc,dc->ac', R, V, W)
                dV = -2 * np.einsum('abd,ac,dc->bc', R, U, W)
                dW = -2 * np.einsum('abd,ac,bc->dc', R, U, V)
                
                # Adam update
                mU = beta1 * mU + (1 - beta1) * dU
                vU = beta2 * vU + (1 - beta2) * dU ** 2
                mU_hat = mU / (1 - beta1 ** t)
                vU_hat = vU / (1 - beta2 ** t)
                U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
                
                mV = beta1 * mV + (1 - beta1) * dV
                vV = beta2 * vV + (1 - beta2) * dV ** 2
                mV_hat = mV / (1 - beta1 ** t)
                vV_hat = vV / (1 - beta2 ** t)
                V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
                
                mW = beta1 * mW + (1 - beta1) * dW
                vW = beta2 * vW + (1 - beta2) * dW ** 2
                mW_hat = mW / (1 - beta1 ** t)
                vW_hat = vW / (1 - beta2 ** t)
                W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
                
                t += 1
                
                if it % 100 == 0 and loss < 1e-6:
                    # Try to round and verify
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    R_check = T - np.einsum('ac,bc,dc->abd', U_r, V_r, W_r)
                    if np.max(np.abs(R_check)) < 1e-6:
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = r
                        return True
                    R_check = T - np.einsum('ac,bc,dc->abd', U, V, W)
                    if np.max(np.abs(R_check)) < 1e-6:
                        best_U, best_V, best_W = U, V, W
                        best_rank = r
                        return True
            
            # After max_iter, check if we got close
            R = T - np.einsum('ac,bc,dc->abd', U, V, W)
            if np.max(np.abs(R)) < 1e-6:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                R_check = T - np.einsum('ac,bc,dc->abd', U_r, V_r, W_r)
                if np.max(np.abs(R_check)) < 1e-6:
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = r
                    return True
                best_U, best_V, best_W = U, V, W
                best_rank = r
                return True
        
        return False
    
    # Try ranks from low to high
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # But for matrix multiplication, the lower bound is higher.
    # For 3x4*4x4, let's try from 16 upward.
    
    for r in range(16, best_rank):
        if time.time() > deadline - 10:
            break
        success = try_rank(r, max_iter=2000, lr=0.05, n_restarts=2)
        if success:
            break
    
    # Convert to lists for the required interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Ensure entries are integer or half-integer
    for mat in [U_list, V_list, W_list]:
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                mat[i][j] = round(mat[i][j] * 2) / 2
    
    # Final verification
    U_np = np.array(U_list)
    V_np = np.array(V_list)
    W_np = np.array(W_list)
    R = T - np.einsum('ac,bc,dc->abd', U_np, V_np, W_np)
    if np.max(np.abs(R)) > 1e-6:
        # Fallback to schoolbook
        U_np, V_np, W_np = schoolbook()
        U_list = U_np.tolist()
        V_list = V_np.tolist()
        W_list = W_np.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
