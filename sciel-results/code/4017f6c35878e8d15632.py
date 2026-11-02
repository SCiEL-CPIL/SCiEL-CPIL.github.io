"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
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

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Reshape to 3D for correct einsum indexing
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # Tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # Expected tensor: T[i,j,k] = 1 for all valid (i,j,k)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)

def _search_rank(target_rank, max_iters=5000, seed=42):
    """Try to find a rank-target_rank decomposition."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    for it in range(max_iters):
        # Random initialization with small values
        U = rng.randn(n * m, target_rank) * 0.1
        V = rng.randn(m * p, target_rank) * 0.1
        W = rng.randn(p * n, target_rank) * 0.1
        
        # Try a few gradient descent steps to see if we can converge
        lr = 0.01
        for step in range(20):
            # Reshape for einsum
            U_3d = U.reshape(n, m, target_rank)
            V_3d = V.reshape(m, p, target_rank)
            W_3d = W.reshape(p, n, target_rank)
            
            # Compute gradient
            T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            expected = np.ones((n, m, p))
            error = T - expected
            
            # Gradient w.r.t. U: dU[i,j,r] = sum_{k} error[i,j,k] * V[j,k,r] * W[k,i,r]
            grad_U_3d = np.einsum('ijk,jkr,kir->ijr', error, V_3d, W_3d)
            grad_V_3d = np.einsum('ijk,ijr,kir->jkr', error, U_3d, W_3d)
            grad_W_3d = np.einsum('ijk,ijr,jkr->kir', error, U_3d, V_3d)
            
            # Reshape gradients back to 2D
            grad_U = grad_U_3d.reshape(n * m, target_rank)
            grad_V = grad_V_3d.reshape(m * p, target_rank)
            grad_W = grad_W_3d.reshape(p * n, target_rank)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        # Check if we converged
        if _verify(U, V, W, tol=1e-7):
            # Try to snap to exact values
            U_exact = np.round(U, 8)
            V_exact = np.round(V, 8)
            W_exact = np.round(W, 8)
            if _verify(U_exact, V_exact, W_exact, tol=1e-9):
                return (U_exact.tolist(), V_exact.tolist(), W_exact.tolist())
            # Return the floating point version
            return (U.tolist(), V.tolist(), W.tolist())
    
    return None

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bounds: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # Try ranks from 25 up to schoolbook_rank - 1
    best_result = None
    best_rank = schoolbook_rank
    
    # Try a few target ranks with limited iterations each
    target_ranks = [25, 26, 27, 28, 30, 32, 35, 40, 45]
    
    for tr in target_ranks:
        if time.time() > deadline - 20:
            break
        result = _search_rank(tr, max_iters=2000, seed=tr * 100 + 42)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, tol=1e-8):
                best_result = (U, V, W)
                best_rank = tr
                break  # Found the lowest rank we tried
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return _build_schoolbook()
# EVOLVE-BLOCK-END
