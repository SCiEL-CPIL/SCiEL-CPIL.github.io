"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check shapes
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # The decomposition should satisfy:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_r U[i*m+j_idx, r] * V[j_idx*p+k_idx, r] * W[k*n+i, r]
    # where for each r, the product U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] 
    # should give 1 if (i,j,k) is a valid term, else 0.
    
    # More precisely, for the tensor T[i,j,k] = 1 (for all valid i,j,k),
    # we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's check this
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_decomposition(n, m, p, target_rank, timeout=100):
    """Try to find a rank decomposition using random search."""
    start_time = time.time()
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
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
        return U, V, W
    
    # Try random initialization and optimization
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(20):
        if time.time() - start_time > timeout - 10:
            break
        
        # Random initialization with structure
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Simple iterative refinement
        for iteration in range(500):
            if time.time() - start_time > timeout - 5:
                break
            
            # For each entry (i,j,k), we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # This is a hard constraint satisfaction problem
            
            # Let's try a different approach: use the fact that we can fix some variables
            # and solve for others
            
            # Actually, let's just check if the random guess is close
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(s - 1.0))
            
            if max_err < 1e-6:
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        
        if best_U is not None:
            break
    
    if best_U is not None:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, try to find a better decomposition
    # Known lower bounds: rank >= max(n*p, m*max(n,p)) = max(12, 16) = 16
    # Schoolbook rank: 48
    
    # Try various ranks from low to high
    for target_rank in range(16, 49):
        if time.time() - start_time > 100:
            break
        
        result = _try_rank_decomposition(n, m, p, target_rank, timeout=100)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
