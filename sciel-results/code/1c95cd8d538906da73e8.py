import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using NumPy broadcasting."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build tensors: U[i,j,c], V[j,k,c], W[k,i,c]
    U_tensor = U.reshape(n, m, rank)
    V_tensor = V.reshape(m, p, rank)
    W_tensor = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Using einsum for efficiency
    T = np.einsum('ijc,jkc,kic->ijk', U_tensor, V_tensor, W_tensor)
    
    # Check if T[i,j,k] = 1 for all i,j,k
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _winograd_5x5_decomposition():
    """
    Winograd-based decomposition for 5x5 matrix multiplication.
    Uses the cyclic group Z_5 convolution structure.
    
    For 5x5 matrix multiplication C = A*B where A is 5x5 and B is 5x5:
    C[i,j] = sum_k A[i,k] * B[k,j]
    
    We can use the fact that this is equivalent to polynomial multiplication
    modulo (x^5 - 1), which can be decomposed using DFT over Z_5.
    
    The rank of the 5x5x5 tensor is at most 5 (using the cyclic convolution structure).
    """
    n = m = p = 5
    
    # The 5-point DFT matrix for Z_5
    # omega = exp(2*pi*i/5)
    # But we need a real-valued decomposition, so we'll use a different approach.
    
    # Actually, let's use a simpler approach: the schoolbook for 5x5 is 125.
    # Winograd can reduce this. Let's try a structured approach.
    
    # For now, return a basic decomposition that we can improve
    rank = 125  # schoolbook
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
    
    return U, V, W, rank

def _construct_4x5x5_decomposition():
    """
    Construct a decomposition for 4x5 * 5x5 matrix multiplication.
    
    The output C is 4x5, where C[i,j] = sum_k A[i,k] * B[k,j]
    A is 4x5, B is 5x5.
    
    We can view this as 4 independent 1x5 * 5x5 multiplications,
    or use a more efficient decomposition.
    """
    n, m, p = 4, 5, 5
    
    # Schoolbook: rank = 4*5*5 = 100
    rank = 100
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
    
    return U, V, W, rank

def _try_winograd_improvement(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition using Winograd-like techniques.
    """
    start_time = time.time()
    
    # Start with schoolbook
    U, V, W, rank = _construct_4x5x5_decomposition()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to reduce rank by finding linear dependencies
    # For 4x5*5x5, the tensor has 4*5*5 = 100 entries
    # We need to find the minimum rank decomposition
    
    # Use gradient-based search to find lower rank decompositions
    for target_rank in range(best_rank - 1, 50, -1):
        if time.time() > deadline - 10:
            break
        
        # Try random initialization and optimization
        for trial in range(3):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(trial * 1000 + target_rank)
            scale = 0.5
            U_try = scale * np.random.randn(n * m, target_rank)
            V_try = scale * np.random.randn(m * p, target_rank)
            W_try = scale * np.random.randn(p * n, target_rank)
            
            # Simple gradient descent
            for it in range(50):
                if time.time() > deadline - 3:
                    break
                
                U_t = U_try.reshape(n, m, target_rank)
                V_t = V_try.reshape(m, p, target_rank)
                W_t = W_try.reshape(p, n, target_rank)
                
                T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
                E = T - 1.0
                loss = np.sum(E * E)
                
                if loss < 1e-8:
                    break
                
                # Gradients
                grad_U = 2.0 * np.einsum('ijk,jkc,kic->ijc', E, V_t, W_t)
                grad_V = 2.0 * np.einsum('ijk,ijc,kic->jkc', E, U_t, W_t)
                grad_W = 2.0 * np.einsum('ijk,ijc,jkc->kic', E, U_t, V_t)
                
                grad_norm = max(np.linalg.norm(grad_U), np.linalg.norm(grad_V), np.linalg.norm(grad_W))
                if grad_norm < 1e-12:
                    break
                
                lr = 0.1 / (grad_norm + 1e-10)
                lr = min(lr, 0.1)
                
                U_try -= lr * grad_U.reshape(-1, target_rank)
                V_try -= lr * grad_V.reshape(-1, target_rank)
                W_try -= lr * grad_W.reshape(-1, target_rank)
            
            if loss < 1e-6:
                # Try snapping to half-integers
                U_snap = np.round(U_try * 2) / 2
                V_snap = np.round(V_try * 2) / 2
                W_snap = np.round(W_try * 2) / 2
                
                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                    best_U, best_V, best_W = U_snap.copy(), V_snap.copy(), W_snap.copy()
                    best_rank = target_rank
                    break
    
    return best_U, best_V, best_W, best_rank

def _snap_to_half_integer(U, V, W):
    """Snap values to the nearest half-integer (0, 0.5, 1, 1.5, ...)."""
    U_snapped = np.round(U * 2) / 2
    V_snapped = np.round(V * 2) / 2
    W_snapped = np.round(W * 2) / 2
    return U_snapped, V_snapped, W_snapped

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    # Start with schoolbook decomposition
    U, V, W, rank_school = _construct_4x5x5_decomposition()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_school
    
    # Try to find better decompositions
    U_try, V_try, W_try, rank_try = _try_winograd_improvement(n, m, p, deadline)
    
    if rank_try < best_rank:
        if _verify_decomposition(U_try, V_try, W_try, n, m, p):
            best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
            best_rank = rank_try
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
