"""Tensor decomposition for 4x4x5 matrix multiplication."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [4, 4, 5]

def construct():
    """
    Find rank-minimal tensor decomposition for 4x4x5 matrix multiplication.
    
    The tensor T[i,j,k] = delta[i,j,k] represents (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    where m=4, n=4, p=5
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T[i,j,k] = 1 if this is a valid multiplication term, else 0
    
    # The matrix multiplication tensor has entries:
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is the standard matrix multiplication tensor
    
    # Strategy: Use a structured decomposition based on known results
    # For 4x4x4, BCL gives rank 19. For 4x4x5, we expect rank ~20-25.
    
    # Let me try a direct algebraic approach using the structure of matrix multiplication.
    # The key insight: we can use bilinear forms to reduce the rank.
    
    # Approach: Try rank 20 first (slightly above the 4x4x4 bound of 19)
    # Use random initialization + gradient-free optimization
    
    best_rank = n * m * p  # Start with schoolbook
    best_decomp = None
    
    # Generate schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try to find lower rank decompositions
    # Use the fact that matrix multiplication tensors have special structure
    
    # Method: Try various structured decompositions
    # For rank r, we need to find U, V, W such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    
    # Let me try a greedy/iterative approach
    # Start with a reasonable rank and try to optimize
    
    target_ranks = [20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 35, 40, 50]
    
    for target_rank in target_ranks:
        if target_rank >= best_rank:
            break
        
        # Try to find a decomposition of this rank
        success = _try_decomposition(n, m, p, target_rank)
        if success:
            U, V, W = success
            best_rank = target_rank
            best_decomp = (U, V, W)
            break
    
    U, V, W = best_decomp
    
    # Convert to lists for the return format
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _try_decomposition(n, m, p, rank, max_iter=1000, tol=1e-8):
    """
    Try to find a rank-`rank` decomposition of the n x m x p matrix multiplication tensor.
    Uses iterative optimization.
    """
    # The tensor T[i,j,k] = 1 for all i,j,k in their respective ranges
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Initialize with random values
    np.random.seed(42)
    
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Normalize to avoid numerical issues
    U /= np.linalg.norm(U)
    V /= np.linalg.norm(V)
    W /= np.linalg.norm(W)
    
    # Use a simple gradient descent approach
    # Loss: sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Compute the current tensor approximation
        # T_approx[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        
        # Compute gradient with respect to U, V, W
        # dL/dU[i*m+j, r] = 2 * sum_{k} (T_approx[i,j,k] - 1) * V[j*p+k, r] * W[k*n+i, r]
        # Similarly for V and W
        
        # For efficiency, compute T_approx first
        T_approx = np.zeros((n, m, p))
        for r in range(rank):
            # U[:, r] is (n*m, 1), V[:, r] is (m*p, 1), W[:, r] is (p*n, 1)
            u_col = U[:, r].reshape(n, m)
            v_col = V[:, r].reshape(m, p)
            w_col = W[:, r].reshape(p, n)
            # T_approx += outer product structure
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_approx[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
        
        # Compute loss
        loss = np.sum((T_approx - 1.0) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        # Compute gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(rank):
            u_col = U[:, r].reshape(n, m)
            v_col = V[:, r].reshape(m, p)
            w_col = W[:, r].reshape(p, n)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        err = T_approx[i, j, k] - 1.0
                        grad_U[i * m + j, r] += 2 * err * v_col[j, k] * w_col[k, i]
                        grad_V[j * p + k, r] += 2 * err * u_col[i, j] * w_col[k, i]
                        grad_W[k * n + i, r] += 2 * err * u_col[i, j] * v_col[j, k]
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Normalize to prevent divergence
        U /= (np.linalg.norm(U) + 1e-10)
        V /= (np.linalg.norm(V) + 1e-10)
        W /= (np.linalg.norm(W) + 1e-10)
    
    # Check if we found a good solution
    if best_loss < 1e-6:
        # Verify the solution
        T_check = np.zeros((n, m, p))
        for r in range(rank):
            u_col = best_U[:, r].reshape(n, m)
            v_col = best_V[:, r].reshape(m, p)
            w_col = best_W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_check[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
        
        if np.max(np.abs(T_check - 1.0)) < 1e-5:
            return best_U, best_V, best_W
    
    return None

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    T_check = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r].reshape(n, m)
        v_col = V[:, r].reshape(m, p)
        w_col = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_check[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
    return np.max(np.abs(T_check - 1.0)) < tol
