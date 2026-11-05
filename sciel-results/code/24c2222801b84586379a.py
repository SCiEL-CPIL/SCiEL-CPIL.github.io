"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)."""
    # T is a 3D array of shape (n, m, p) where T[i,j,k] = 1
    # But we need to think of it as a tensor that we're decomposing
    # The multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] such that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # Actually T[i,j,k] = 1 for all i,j,k in the standard case
    return np.ones((n, m, p), dtype=float)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Build the reconstructed tensor
    # T_reconstructed[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For efficiency, we can check a few entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False, val, (i,j,k)
    return True, None, None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_decomposition(n, m, p, rank, max_time=30):
    """Try to find a rank decomposition using random initialization and iterative refinement."""
    start_time = time.time()
    np.random.seed(42)
    
    best_result = None
    best_error = float('inf')
    
    for attempt in range(50):
        if time.time() - start_time > max_time:
            break
            
        # Random initialization with sparse structure
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Iterative refinement: fix two, optimize the third
        for iteration in range(200):
            if time.time() - start_time > max_time:
                break
            
            # Update U: for each (i,j), U[i*m+j, :] should satisfy
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # This is a linear system for each (i,j)
            for i in range(n):
                for j in range(m):
                    # Build system: for each k, sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
                    A_mat = np.zeros((p, rank))
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve A_mat @ U[i*m+j, :] = ones(p)
                    try:
                        U[i*m+j, :], residuals, rank_s, sv = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                    except:
                        pass
            
            # Update V: for each (j,k), V[j*p+k, :] should satisfy
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, rank))
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :], residuals, rank_s, sv = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                    except:
                        pass
            
            # Update W: for each (k,i), W[k*n+i, :] should satisfy
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, rank))
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :], residuals, rank_s, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                    except:
                        pass
        
        # Check error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    error += (val - 1.0) ** 2
        
        if error < best_error:
            best_error = error
            best_result = (U.copy(), V.copy(), W.copy())
        
        if best_error < 1e-6:
            break
    
    if best_result is not None and best_error < 1e-6:
        U, V, W = best_result
        # Convert to list of lists with rounded values
        U_list = [[round(U[i][j], 6) for j in range(rank)] for i in range(n*m)]
        V_list = [[round(V[i][j], 6) for j in range(rank)] for i in range(m*p)]
        W_list = [[round(W[i][j], 6) for j in range(rank)] for i in range(p*n)]
        
        # Verify
        valid, _, _ = _verify_decomposition(U_list, V_list, W_list, n, m, p)
        if valid:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Leave margin before 150s limit
    
    # Try decreasing ranks from a reasonable upper bound
    # Schoolbook is 45. Try to find something better.
    # Known: for 3x3x3, best is 23. For 3x3x5, we can try ranks around 30-40.
    
    # First, get the schoolbook as fallback
    fallback = _schoolbook(n, m, p)
    
    # Try to find a better decomposition
    for rank in range(40, 45):
        if time.time() > deadline - 30:
            break
        result = _try_rank_decomposition(n, m, p, rank, max_time=20)
        if result is not None:
            return result
    
    # Try rank 45 (schoolbook) as final fallback
    return fallback
# EVOLVE-BLOCK-END
