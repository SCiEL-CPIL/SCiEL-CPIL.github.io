"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)  
    # W shape: (p*n, rank)
    # The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication: T[i,j,k] = 1 if this is a valid (i,j,k) triple
    # Actually for matrix multiplication A*B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k (since each product contributes)
    # Wait, let me reconsider. The matrix multiplication tensor T[i,j,k] = delta_{i,i} * delta_{k,k} = 1
    # No, that's not right either.
    
    # The matrix multiplication tensor is defined as:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is because C[i,k] = sum_j A[i,j] B[j,k], and each term A[i,j]B[j,k] 
    # contributes to the product with coefficient 1.
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    n, m, p = 2, 3, 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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

def _try_reduce(n, m, p, target_rank, max_iter=500, deadline=None):
    """Try to find a decomposition of rank target_rank using random search."""
    if deadline is not None and time.time() > deadline:
        return None
    
    np.random.seed(42)
    U = np.random.randn(n * m, target_rank) * 0.5
    V = np.random.randn(m * p, target_rank) * 0.5
    W = np.random.randn(p * n, target_rank) * 0.5
    
    # Try a simple gradient-free approach: random restarts with local refinement
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Compute current error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        err = np.sqrt(err / (n*m*p))
        
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-10:
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
        # Random perturbation
        scale = 0.1 * (1 - it / max_iter)
        U += np.random.randn(n * m, target_rank) * scale
        V += np.random.randn(m * p, target_rank) * scale
        W += np.random.randn(p * n, target_rank) * scale
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # Start with schoolbook
    schoolbook = _build_schoolbook(n, m, p)
    
    # Try to reduce rank
    # Known lower bounds: for 2x3x7, the minimum rank is at least max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # But actual minimum is likely higher. Let's try various ranks.
    
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try ranks from lower bound up
    min_possible = max(n*m, m*p, p*n)  # = 21
    for rank in range(min_possible, best_rank):
        if time.time() > deadline:
            break
        
        result = _try_reduce(n, m, p, rank, max_iter=100, deadline=deadline)
        if result is not None:
            # Verify
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                best_result = result
                best_rank = rank
                break  # Found a valid lower rank, stop
    
    return best_result
# EVOLVE-BLOCK-END
