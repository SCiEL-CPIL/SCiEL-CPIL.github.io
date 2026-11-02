"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    # Check that U, V, W have correct shapes
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    # Check the tensor equation: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _als_decompose(n, m, p, rank, max_iter=1000, tol=1e-10, seed=42):
    """Use ALS to find a rank-r decomposition."""
    np.random.seed(seed)
    # Initialize
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1
    # We need: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = float('inf')
    
    for it in range(max_iter):
        # Update U
        for i in range(n):
            for j in range(m):
                # For fixed i,j, we need: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 for all k
                # This is a linear system in U[i*m+j,:]
                # Let A[k,t] = V[j*p+k,t] * W[k*n+i,t]
                # Then sum_t U[i*m+j,t] * A[k,t] = 1 for all k
                A = np.zeros((p, rank))
                for k in range(p):
                    for t in range(rank):
                        A[k, t] = V[j*p+k, t] * W[k*n+i, t]
                # Solve A^T * U[i*m+j,:] = 1 (least squares)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                # For fixed j,k, we need: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 for all i
                A = np.zeros((n, rank))
                for i in range(n):
                    for t in range(rank):
                        A[i, t] = U[i*m+j, t] * W[k*n+i, t]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for i in range(n):
            for k in range(p):
                # For fixed i,k, we need: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 for all j
                A = np.zeros((m, rank))
                for j in range(m):
                    for t in range(range(rank) if False else range(rank)):
                        A[j, t] = U[i*m+j, t] * V[j*p+k, t]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A.T, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    err += (s - 1.0)**2
        err = np.sqrt(err / (n*m*p))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    return best_U, best_V, best_W, best_err

def _try_rank(n, m, p, rank, max_iter=500, n_restarts=5):
    """Try to find a valid decomposition of given rank."""
    best_result = None
    best_err = float('inf')
    
    for restart in range(n_restarts):
        U, V, W, err = _als_decompose(n, m, p, rank, max_iter=max_iter, seed=restart)
        if err < best_err:
            best_err = err
            best_result = (U, V, W)
        if err < 1e-10:
            break
    
    if best_result is not None:
        U, V, W = best_result
        # Verify
        U_list = [[float(U[i,j]) for j in range(rank)] for i in range(n*m)]
        V_list = [[float(V[i,j]) for j in range(rank)] for i in range(m*p)]
        W_list = [[float(W[i,j]) for j in range(rank)] for i in range(p*n)]
        if _verify(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Start with schoolbook
    best = _schoolbook()
    
    # Try to find lower rank decompositions
    # Known lower bounds suggest we can do better than 90
    # Try ranks from low to high
    for rank in range(30, 90):
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, rank, max_iter=300, n_restarts=3)
        if result is not None:
            best = result
            break  # Found the lowest rank that works
    
    return best
# EVOLVE-BLOCK-END
