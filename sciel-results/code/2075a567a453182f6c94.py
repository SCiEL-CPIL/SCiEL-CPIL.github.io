"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0  # safety margin before 150s kill
    
    # Build schoolbook decomposition
    def build_schoolbook():
        terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for c, (i, j, k) in enumerate(terms):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W

    # Verify decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    def verify(U, V, W, tol=1e-9):
        # Vectorized verification
        # Reshape to 3D tensors
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        
        # Compute sum over rank dimension
        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # This is a tensor contraction.
        # We can do this by iterating over c or using einsum.
        # einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        try:
            T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            if np.max(np.abs(T - 1.0)) > tol:
                return False
            return True
        except Exception:
            return False

    # Compute tensor error (sum of squared errors)
    def tensor_sse(U, V, W):
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        return np.sum((T - 1.0) ** 2)

    # Vectorized Gradient Descent Step
    def grad_step(U, V, W, lr):
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        
        # Error tensor E[i,j,k] = T[i,j,k] - 1
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        E = T - 1.0
        
        # Gradients
        # dE/dU[i,j,c] = sum_k V[j,k,c] W[k,i,c] E[i,j,k]
        # dE/dV[j,k,c] = sum_i U[i,j,c] W[k,i,c] E[i,j,k]
        # dE/dW[k,i,c] = sum_j U[i,j,c] V[j,k,c] E[i,j,k]
        
        G_U = np.einsum('jkc,kic,ijk->ijc', V_3d, W_3d, E)
        G_V = np.einsum('ijc,kic,ijk->jkc', U_3d, W_3d, E)
        G_W = np.einsum('ijc,jkc,ijk->kic', U_3d, V_3d, E)
        
        # Update
        U_new = U - lr * G_U.reshape(n*m, -1)
        V_new = V - lr * G_V.reshape(m*p, -1)
        W_new = W - lr * G_W.reshape(p*n, -1)
        
        return U_new, V_new, W_new

    # Start with schoolbook
    U, V, W = build_schoolbook()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if not verify(best_U, best_V, best_W):
        raise ValueError("Schoolbook decomposition failed verification")
    
    np.random.seed(42)
    
    # Strategy: Try to reduce rank using vectorized gradient descent
    # We start from schoolbook and try to prune columns or optimize at lower ranks.
    
    # First, try to optimize the schoolbook decomposition to see if we can reduce error (should be 0)
    # Then try to zero out columns.
    
    # A better strategy: 
    # 1. Try ranks from 41 down to 30.
    # 2. For each rank, initialize by taking the first R columns of schoolbook and adding noise.
    # 3. Run vectorized gradient descent.
    
    for target_rank in range(41, 30, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(5): # Fewer attempts since each is faster
            if time.time() > deadline - 10:
                break
            
            # Initialize
            U_c = np.zeros((n * m, target_rank))
            V_c = np.zeros((m * p, target_rank))
            W_c = np.zeros((p * n, target_rank))
            
            terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
            for c in range(target_rank):
                i, j, k = terms[c]
                U_c[i * m + j, c] = 1.0
                V_c[j * p + k, c] = 1.0
                W_c[k * n + i, c] = 1.0
            
            # Add noise
            U_c += np.random.randn(n * m, target_rank) * 0.1
            V_c += np.random.randn(m * p, target_rank) * 0.1
            W_c += np.random.randn(p * n, target_rank) * 0.1
            
            # Optimize
            best_sse = tensor_sse(U_c, V_c, W_c)
            lr = 0.01
            for it in range(500):
                if time.time() > deadline - 10:
                    break
                U_c, V_c, W_c = grad_step(U_c, V_c, W_c, lr)
                sse = tensor_sse(U_c, V_c, W_c)
                if sse < best_sse:
                    best_sse = sse
                else:
                    # Reduce learning rate if stuck
                    lr *= 0.5
                    if lr < 1e-6:
                        break
            
            # Check validity
            if best_sse < 1e-9:
                if verify(U_c, V_c, W_c):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_c.copy(), V_c.copy(), W_c.copy()
                    found = True
                    break
            
            # Try snapping to half-integers
            U_snap = np.round(U_c * 2.0) / 2.0
            V_snap = np.round(V_c * 2.0) / 2.0
            W_snap = np.round(W_c * 2.0) / 2.0
            if verify(U_snap, V_snap, W_snap):
                best_rank = target_rank
                best_U, best_V, best_W = U_snap.copy(), V_snap.copy(), W_snap.copy()
                found = True
                break
        
        if found:
            continue
        else:
            if target_rank <= 38:
                break
    
    # Final verification
    if not verify(best_U, best_V, best_W):
        best_U, best_V, best_W = build_schoolbook()
        best_rank = n * m * p
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
