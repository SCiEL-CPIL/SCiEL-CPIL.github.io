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
        # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Note: W is indexed by k, i in the tensor sum, but stored as (p*n, rank) -> (p, n, rank)
        # The term in sum is W[k, i, r].
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        
        # Check all entries are 1
        if not np.allclose(T, 1.0, atol=1e-8):
            return False
        return True
    
    def try_rank_reduction(rank):
        """Try to find a rank-reduction using random initialization and iterative refinement."""
        if rank >= schoolbook_rank:
            return None
        
        best_U, best_V, best_W = None, None, None
        best_error = float('inf')
        
        # Reduced search budget to ensure we finish in time
        n_attempts = 10
        iters_per_attempt = 100
        
        for attempt in range(n_attempts):
            if _time_left() < 10.0:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            for it in range(iters_per_attempt):
                if _time_left() < 10.0:
                    break
                
                # Reshape for einsum
                U_r = U.reshape(n, m, rank)
                V_r = V.reshape(m, p, rank)
                W_r = W.reshape(p, n, rank)
                
                # Compute T
                T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                E = 1.0 - T
                
                error = np.sum(E * E)
                if error < best_error:
                    best_error = error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    if error < 1e-10:
                        break
                
                # Compute gradients
                # dU[i,j,r] = -2 * sum_k E[i,j,k] V[j,k,r] W[k,i,r]
                dU_r = -2.0 * np.einsum('ijk,jkr,kir->ijr', E, V_r, W_r)
                # dV[j,k,r] = -2 * sum_i E[i,j,k] U[i,j,r] W[k,i,r]
                dV_r = -2.0 * np.einsum('ijk,ijr,kir->jkr', E, U_r, W_r)
                # dW[k,i,r] = -2 * sum_j E[i,j,k] U[i,j,r] V[j,k,r]
                dW_r = -2.0 * np.einsum('ijk,ijr,jkr->kir', E, U_r, V_r)
                
                # Flatten gradients
                dU = dU_r.reshape(n * m, rank)
                dV = dV_r.reshape(m * p, rank)
                dW = dW_r.reshape(p * n, rank)
                
                # Gradient descent step with adaptive learning rate
                lr = 0.01
                # Simple line search or just fixed small step
                U += lr * dU
                V += lr * dV
                W += lr * dW
            
            if best_error < 1e-10:
                break
        
        if best_error < 1e-6:
            return best_U, best_V, best_W
        return None
    
    # Try to reduce rank from 47 down
    U, V, W = schoolbook()
    
    # Only try a few ranks to stay within time limit
    for target_rank in [schoolbook_rank - 1, schoolbook_rank - 2]:
        if _time_left() < 20.0:
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
