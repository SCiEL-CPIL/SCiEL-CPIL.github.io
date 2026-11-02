"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    deadline = time.time() + 145  # 5s safety margin for 150s limit

    # Build the matrix multiplication tensor as a matrix
    # Tensor T[i,j,k] = delta_{i,j,k} for matmul (A is n x m, B is m x p, C is p x n)
    # Actually for matmul: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if this is a valid multiplication term, else 0
    # The tensor is a 3-way tensor with T[i,j,k] = 1 for all valid (i,j,k)
    # But wait, in matmul the tensor is defined as T[i,j,k] = 1 if we use A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k in their ranges.
    
    # The matrix multiplication tensor has entries T[i,j,k] = 1 for all i,j,k
    # We want to find rank-r decomposition: T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
    
    # U has shape (n, r), V has shape (m, r), W has shape (p, r)
    # We need: for all i,j,k: sum_c U[i,c]*V[j,c]*W[k,c] = 1
    
    # This is a linear system in the products, but nonlinear in U,V,W
    # For a given rank r, we can try to solve using random initialization
    
    # Try ranks from a lower bound up to schoolbook
    # Known lower bound: max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # Also rank >= max(n, m, p) = 10
    
    # Let's try to find a decomposition with rank < 60
    # We'll use a random search approach with verification
    
    def verify(U_mat, V_mat, W_mat, tol=1e-10):
        """Verify that U @ V^T @ W^T approximates the all-ones tensor"""
        # Compute T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
        # This is equivalent to: for each c, the outer product U[:,c] @ V[:,c]^T @ W[:,c]^T
        # Sum over c
        # Efficient computation:
        # T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
        # = sum_c (U[i,c]*W[k,c]) * V[j,c]
        # Let's compute it as: for each i,k, T[i,:,k] = V @ (U[i,:] * W[k,:])^T
        # Or: T = sum_c np.einsum('ic,jc,kc->ijk', U_mat, V_mat, W_mat)
        T = np.einsum('ic,jc,kc->ijk', U_mat, V_mat, W_mat)
        target = np.ones((n, m, p))
        return np.allclose(T, target, atol=tol)

    def try_random_decomposition(rank, n_trials=100, seed=42):
        """Try to find a rank-r decomposition using random initialization and verification"""
        rng = np.random.RandomState(seed)
        for trial in range(n_trials):
            # Random initialization
            U_mat = rng.randn(n, rank)
            V_mat = rng.randn(m, rank)
            W_mat = rng.randn(p, rank)
            
            # Simple verification
            if verify(U_mat, V_mat, W_mat, tol=1e-8):
                return U_mat, V_mat, W_mat
        return None

    # Try ranks below schoolbook
    # Start from a reasonable lower bound
    lower_bound = max(n * m, m * p, p * n)
    
    # Try a few ranks below 60
    for rank in range(lower_bound, rank_schoolbook):
        if time.time() > deadline:
            break
        result = try_random_decomposition(rank, n_trials=50)
        if result is not None:
            U_mat, V_mat, W_mat = result
            # Convert to the required format
            # U should be (n*m, rank) with U[i*m+j, c] = U_mat[i, c]
            # Wait, the task says U has shape (n*m, rank)
            # And the tensor is for multiplying 2x3 by 3x10
            # So the first matrix is 2x3 (n x m), second is 3x10 (m x p)
            # Output is 2x10 (n x p)
            # The tensor T[i,j,k] where i is row of first mat, j is col of first/row of second, k is col of second
            # U[i*m+j, c] corresponds to A[i,j]
            # V[j*p+k, c] corresponds to B[j,k]
            # W[k*n+i, c] corresponds to C[k,i] or C[i,k]?
            # The task says "output factor follows the official trace convention W[k*n+i,r]"
            # So W[k*n+i, c] is the coefficient for C[i,k]
            
            # Build U_full: shape (n*m, rank)
            U_full = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    U_full[i * m + j] = U_mat[i]
            
            # Build V_full: shape (m*p, rank)
            V_full = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    V_full[j * p + k] = V_mat[j]
            
            # Build W_full: shape (p*n, rank)
            W_full = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    W_full[k * n + i] = W_mat[k]
            
            # Verify the full decomposition
            # C[i,k] = sum_j sum_c U_full[i*m+j, c] * V_full[j*p+k, c] * W_full[k*n+i, c]
            # Should equal A[i,:] @ B[:,k]
            
            # Convert to lists
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback to schoolbook
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
