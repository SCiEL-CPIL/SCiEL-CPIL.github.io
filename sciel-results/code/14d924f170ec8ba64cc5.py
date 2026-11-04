"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _compute_error(U, V, W, n, m, p):
    """Compute the error of the decomposition."""
    rank = U.shape[1]
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                error += (s - 1.0)**2
    return np.sqrt(error)

def _als_decomposition(n, m, p, rank, num_iters=20, tol=1e-12, seed=None):
    """
    Use Alternating Least Squares to find a rank-r decomposition.
    Returns (U, V, W) as numpy arrays, or None if it fails.
    """
    if seed is not None:
        np.random.seed(seed)
    
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    prev_error = float('inf')
    
    for it in range(num_iters):
        # Update U: for each (i,j)
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                b = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k)
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                b = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(B, b, rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i)
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(C, b, rcond=None)[0]
                except:
                    pass
        
        error = _compute_error(U, V, W, n, m, p)
        
        if error < tol:
            break
        if abs(error - prev_error) < 1e-15:
            break
        prev_error = error
    
    if error < tol:
        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120 - 15  # 105s for safety
    
    result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from 89 down to 85
    for rank in range(best_rank - 1, 84, -1):
        if time.time() - start_time > deadline:
            break
        
        found = False
        for seed in range(3):
            if time.time() - start_time > deadline:
                break
            
            res = _als_decomposition(n, m, p, rank, num_iters=20, tol=1e-12, seed=seed)
            if res is not None:
                U, V, W = res
                # Convert to lists and verify
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            continue
    
    return result
# EVOLVE-BLOCK-END

