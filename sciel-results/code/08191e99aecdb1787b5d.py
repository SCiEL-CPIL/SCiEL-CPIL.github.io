import numpy as np
import time
import random

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m + j, j*p + k, k*n + i] = 1 for all valid i,j,k
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Start with schoolbook rank
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Initialize schoolbook
    U0 = np.zeros((n * m, best_rank))
    V0 = np.zeros((m * p, best_rank))
    W0 = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1
    best_U, best_V, best_W = U0, V0, W0
    
    # Try to reduce rank using randomized local search
    # Strategy: try to find a lower-rank decomposition by starting from random
    # rank-r candidates and using gradient-like updates
    
    def compute_residual(U, V, W, T):
        """Compute ||T - sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]||_F^2"""
        # Efficient computation: for each column r, compute the outer product
        # and accumulate
        residual_sq = 0.0
        for r in range(U.shape[1]):
            # u ⊗ v ⊗ w flattened
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            # The contribution is u[i]*v[j]*w[k] for all i,j,k
            # We need to subtract from T
            # Compute the 3D tensor contribution
            contrib = np.einsum('i,j,k->ijk', u, v, w)
            residual_sq += np.sum((T - contrib) ** 2)
        # Actually we need the full residual, let's compute it properly
        recon = np.zeros_like(T)
        for r in range(U.shape[1]):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            recon += np.einsum('i,j,k->ijk', u, v, w)
        return np.sum((T - recon) ** 2)
    
    def try_reduce_rank(target_rank, max_iters=500, lr=0.01):
        """Try to find a decomposition of given target_rank"""
        if time.time() - start_time > 130:
            return None
        
        # Random initialization
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() - start_time > 130:
                return None
            
            # Compute residual
            recon = np.zeros_like(T)
            for r in range(target_rank):
                recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            err = np.sum((T - recon) ** 2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_err < 1e-10:
                    return best_U, best_V, best_W
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r in range(target_rank):
                u = U[:, r]
                v = V[:, r]
                w = W[:, r]
                contrib = np.einsum('i,j,k->ijk', u, v, w)
                res = T - recon
                
                # Gradient w.r.t. u: sum over j,k of res[i,j,k] * v[j] * w[k]
                grad_u = np.einsum('ijk,j,k->i', res, v, w)
                grad_v = np.einsum('ijk,i,k->j', res, u, w)
                grad_w = np.einsum('ijk,i,j->k', res, u, v)
                
                grad_U[:, r] = grad_u
                grad_V[:, r] = grad_v
                grad_W[:, r] = grad_w
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Decay learning rate
            lr *= 0.99
        
        if best_err < 1e-8:
            return best_U, best_V, best_W
        return None
    
    # Try various target ranks from schoolbook down
    for target_rank in range(best_rank - 1, max(0, best_rank - 20), -1):
        if time.time() - start_time > 130:
            break
        
        result = try_reduce_rank(target_rank, max_iters=300, lr=0.05)
        if result is not None:
            U, V, W = result
            # Verify
            recon = np.zeros_like(T)
            for r in range(target_rank):
                recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            if np.max(np.abs(T - recon)) < 1e-8:
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
    
    # Convert to lists with rounded values
    def to_list(arr):
        # Round to nearest 0.5 or integer if close
        rounded = np.round(arr * 2) / 2
        # Check if it's close to the original
        if np.max(np.abs(arr - rounded)) < 1e-6:
            return rounded.tolist()
        else:
            return arr.tolist()
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
