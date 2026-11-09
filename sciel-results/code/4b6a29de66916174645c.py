import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def _make_tensor(n, m, p):
    """Create the matrix multiplication tensor T[i,j,k] = 1 for all valid indices."""
    return np.ones((n, m, p))

def _als_decompose(n, m, p, rank, max_iter=200, tol=1e-10, seed=0):
    """ALS decomposition of the all-ones tensor with matrix-multiplication index mapping."""
    rng = np.random.RandomState(seed)
    
    # Initialize factors randomly
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Update U: for each (i,j), U[i*m+j, :] = solve(V * W contracted appropriately)
        # The equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # For fixed (i,j): sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
        # This is: U[i*m+j, :] @ (V[j*p:, :] * W[i::n, :]) = ones(p)
        # where V[j*p+k, r] * W[k*n+i, r] forms a (p, rank) matrix
        
        # Build the system for U
        for i in range(n):
            for j in range(m):
                # M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve: U[i*m+j, :] @ M = ones(p)
                # This is M.T @ U[i*m+j, :] = ones(p)
                # U[i*m+j, :] = pinv(M.T) @ ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                except:
                    U[i*m+j, :] = np.linalg.pinv(M.T) @ np.ones(p)
        
        # Update V: for each (j,k), V[j*p+k, :] = solve
        # For fixed (j,k): sum_r V[j*p+k, r] * U[i*m+j, r] * W[k*n+i, r] = 1 for all i
        # M[i, r] = U[i*m+j, r] * W[k*n+i, r]
        # V[j*p+k, :] @ M = ones(n)
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                except:
                    V[j*p+k, :] = np.linalg.pinv(M.T) @ np.ones(n)
        
        # Update W: for each (k,i), W[k*n+i, :] = solve
        # For fixed (k,i): sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for all j
        # M[j, r] = U[i*m+j, r] * V[j*p+k, r]
        # W[k*n+i, :] @ M = ones(m)
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                except:
                    W[k*n+i, :] = np.linalg.pinv(M.T) @ np.ones(m)
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    return best_U, best_V, best_W, best_err

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    best_result = None
    best_rank = n * m * p  # Schoolbook rank as fallback
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    # But for matrix multiplication, better lower bounds exist
    # For 2x3 x 3x5, let's try from rank 12 upward
    
    min_rank = max(n*m, m*p, p*n)  # Basic lower bound
    # Try a range of ranks
    for rank in range(min_rank, n*m*p + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts for this rank
        max_restarts = 10
        if time.time() > deadline:
            break
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            U, V, W, err = _als_decompose(n, m, p, rank, max_iter=100, tol=1e-12, seed=restart * 1000 + rank)
            
            if _verify(U, V, W, n, m, p, tol=1e-8):
                # Convert to lists with integer/half-integer entries if possible
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Fallback to schoolbook
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
