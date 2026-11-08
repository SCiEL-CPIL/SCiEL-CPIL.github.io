"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 by 4x7 matrix.
    Uses an iterative tensor rank minimization approach with sparsification.
    """
    start_time = time.time()
    deadline = 135  # Leave 15s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 112
    
    # Initialize with schoolbook decomposition
    best_rank = rank_schoolbook
    best_U = np.zeros((n * m, rank_schoolbook))
    best_V = np.zeros((m * p, rank_schoolbook))
    best_W = np.zeros((p * n, rank_schoolbook))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, column] = 1
                best_V[j * p + k, column] = 1
                best_W[k * n + i, column] = 1
                column += 1
    
    # Try to reduce rank through iterative optimization
    # Strategy: Start with random low-rank initialization and refine
    
    np.random.seed(42)
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # should equal 1 if i,j,k form a valid multiplication term, else 0
        # Actually, for matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
        # Tensor T[i,j,k] = delta (it's 1 for all valid (i,j,k), 0 otherwise)
        # Wait, the standard matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
        # because C[i,k] = sum_j A[i,j] * B[j,k]
        
        # Let's verify by checking if the decomposition reproduces the tensor
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # Build the full tensor from decomposition
        T_reconstructed = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    T_reconstructed[i, j, k] = val
        
        # The target tensor is all ones (matrix multiplication tensor)
        T_target = np.ones((n, m, p))
        
        return np.allclose(T_reconstructed, T_target, atol=tol)
    
    def try_reduce_rank(current_U, current_V, current_W, target_rank, max_iter=5000):
        """Try to reduce the rank to target_rank using iterative refinement."""
        if target_rank >= current_U.shape[1]:
            return None
        
        # Initialize with random matrices of target rank
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Scale to help convergence
        U /= np.sqrt(target_rank)
        V /= np.sqrt(target_rank)
        W /= np.sqrt(target_rank)
        
        lr = 0.01  # Learning rate
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute gradients
            # T_reconstructed[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # Target T[i,j,k] = 1
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Current value
                        val = 0.0
                        for c in range(target_rank):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        
                        error = 1.0 - val  # Target is 1
                        
                        for c in range(target_rank):
                            grad_U[i * m + j, c] += error * V[j * p + k, c] * W[k * n + i, c]
                            grad_V[j * p + k, c] += error * U[i * m + j, c] * W[k * n + i, c]
                            grad_W[k * n + i, c] += error * U[i * m + j, c] * V[j * p + k, c]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if iteration % 100 == 0:
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(target_rank):
                                val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            max_err = max(max_err, abs(1.0 - val))
                
                if max_err < 1e-8:
                    # Try to sparsify
                    return _sparsify(U, V, W, target_rank)
        
        return None
    
    def _sparsify(U, V, W, rank):
        """Try to sparsify the decomposition to get integer/half-integer entries."""
        # Round to nearest 0.5
        U_sp = np.round(U * 2) / 2
        V_sp = np.round(V * 2) / 2
        W_sp = np.round(W * 2) / 2
        
        if verify_decomposition(U_sp, V_sp, W_sp, tol=1e-6):
            return U_sp, V_sp, W_sp
        
        # Try rounding to integers
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        
        if verify_decomposition(U_int, V_int, W_int, tol=1e-6):
            return U_int, V_int, W_int
        
        return None
    
    # Try various target ranks, starting from lower
    # Known results suggest rank around 40-50 might be achievable
    for target_rank in range(40, min(rank_schoolbook, 80), 2):
        if time.time() > deadline:
            break
        
        result = try_reduce_rank(best_U, best_V, best_W, target_rank, max_iter=2000)
        if result is not None:
            U_new, V_new, W_new = result
            if U_new.shape[1] < best_rank:
                best_rank = U_new.shape[1]
                best_U, best_V, best_W = U_new, V_new, W_new
                print(f"Found decomposition with rank {best_rank}")
    
    # If no better found, return schoolbook
    if best_rank == rank_schoolbook:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
