"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook():
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
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the matrix multiplication tensor
    # T[i*m+j, j'*p+k, k'*n+i'] should equal delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # Actually, the tensor is: T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a, b, c] = 1 if i==i', j==j', k==k' else 0
    # And the decomposition is: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    
    # Reconstruct T from decomposition
    # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Check on a sample of entries
    # Full tensor has size (n*m) x (m*p) x (p*n) = 15 x 30 x 18 = 8100 entries
    # We can check all of them
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build true T
    T_true = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m+j, j*p+k, k*n+i] = 1.0
    
    diff = np.max(np.abs(T_recon - T_true))
    return diff < tol

def construct():
    start_time = time.time()
    deadline = 140  # Leave 10s margin
    
    # First, try to find a better decomposition via structured search
    # Strategy: start with schoolbook, then try to merge terms
    
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook()
    best = schoolbook
    best_rank = n * m * p
    
    # Try random perturbations and rank reduction
    np.random.seed(42)
    
    # Approach: try to find a rank r < 90 decomposition
    # Use the fact that the tensor lies in a specific variety
    
    # For now, return schoolbook as fallback
    # Try a quick search for rank 85-89
    for target_rank in range(89, 80, -1):
        if time.time() > deadline:
            break
        # Try random initialization
        for attempt in range(20):
            if time.time() > deadline:
                break
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Simple gradient descent for a few iterations
            # Loss = ||T_recon - T_true||_F^2
            lr = 0.01
            for it in range(50):
                if time.time() > deadline:
                    break
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                # Gradient w.r.t. U: grad_U[a,r] = sum_{b,c} (T_recon[a,b,c] - T_true[a,b,c]) * V[b,r] * W[c,r]
                residual = T_recon - _get_true_tensor()
                grad_U = np.einsum('abc,br,cr->ar', residual, V, W)
                grad_V = np.einsum('abc,ar,cr->br', residual, U, W)
                grad_W = np.einsum('abc,ar,br->cr', residual, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if it % 10 == 0:
                    loss = np.sum(residual**2)
                    if loss < 1e-8:
                        break
            
            if time.time() > deadline:
                break
            
            # Check if we found a valid decomposition
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                break
        if best_rank < n*m*p:
            break
    
    if time.time() > deadline:
        return schoolbook
    
    return best

def _get_true_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T
# EVOLVE-BLOCK-END
