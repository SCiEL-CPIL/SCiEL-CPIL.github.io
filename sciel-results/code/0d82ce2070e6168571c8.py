"""Matrix multiplication 3,4,4 rank minimization."""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

TIME_LIMIT = 145.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _time_left():
    return TIME_LIMIT - _elapsed()

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Build schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return U, V, W
    
    def verify(U, V, W):
        """Check if the decomposition exactly reproduces the matrix multiplication tensor.
        T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1 for all (i,j,k)."""
        rank = U.shape[1]
        # Build the full tensor efficiently
        # T has shape (n, m, p)
        # For each r, the contribution is outer(U[:,r], V[:,r], W[:,r]) reshaped
        # Actually T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Let's compute this directly
        T = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            # This is a 3D outer product-like operation
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        # Check all entries are 1
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True
    
    def try_rank_reduction(rank):
        """Try to find a rank-reduction using random initialization and iterative refinement."""
        if rank >= schoolbook_rank:
            return None
        
        # We need to find U (n*m x rank), V (m*p x rank), W (p*n x rank)
        # such that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
        
        # Use random initialization with small values
        # and try to optimize
        
        best_U, best_V, best_W = None, None, None
        best_error = float('inf')
        
        n_attempts = 50
        iters_per_attempt = 200
        
        for attempt in range(n_attempts):
            if _time_left() < 5.0:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            for it in range(iters_per_attempt):
                if _time_left() < 5.0:
                    break
                
                # Compute current tensor and gradient
                T = np.zeros((n, m, p), dtype=np.float64)
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for r in range(rank):
                    u = U[:, r].reshape(n, m)
                    v = V[:, r].reshape(m, p)
                    w = W[:, r].reshape(p, n)
                    # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                    # Gradient w.r.t u[i,j]: sum_k v[j,k] * w[k,i]
                    # Gradient w.r.t v[j,k]: sum_i u[i,j] * w[k,i]
                    # Gradient w.r.t w[k,i]: sum_j u[i,j] * v[j,k]
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = u[i, j] * v[j, k] * w[k, i]
                                T[i, j, k] += val
                                # Error = 1 - T[i,j,k]
                                err = 1.0 - T[i, j, k]
                                dU[i * m + j, r] += err * v[j, k] * w[k, i]
                                dV[j * p + k, r] += err * u[i, j] * w[k, i]
                                dW[k * n + i, r] += err * u[i, j] * v[j, k]
                
                error = np.sum((T - 1.0) ** 2)
                if error < best_error:
                    best_error = error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    if error < 1e-9:
                        break
                
                # Gradient descent step
                lr = 0.01
                U += lr * dU
                V += lr * dV
                W += lr * dW
            
            if best_error < 1e-9:
                break
        
        if best_error < 1e-6:
            return best_U, best_V, best_W
        return None
    
    # Try to reduce rank from 47 down
    U, V, W = schoolbook()
    
    for target_rank in range(schoolbook_rank - 1, max(30, schoolbook_rank - 10), -1):
        if _time_left() < 10.0:
            break
        
        result = try_rank_reduction(target_rank)
        if result is not None:
            U, V, W = result
            break
    
    # Final verification
    if not verify(U, V, W):
        # Fallback to schoolbook
        U, V, W = schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
