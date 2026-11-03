import time
import numpy as np
from scipy import linalg

DIMENSIONS = [3, 4, 4]
_START_TIME = time.time()
_TIMEOUT = 115  # safety margin before 120s limit

def _schoolbook():
    """Return schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _tensor_error(U, V, W, n, m, p):
    """Compute max absolute error of the tensor decomposition."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.max(np.abs(T - 1.0))

def _als_iteration(U, V, W, n, m, p):
    """Perform one alternating least squares iteration.
    U, V, W have shape (n*m, r), (m*p, r), (p*n, r) respectively.
    Returns new U, V, W.
    """
    r = U.shape[1]
    U_new = np.zeros_like(U)
    V_new = np.zeros_like(V)
    W_new = np.zeros_like(W)

    # Solve for U
    for i in range(n):
        for j in range(m):
            # For fixed (i,j), we need:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # This is a linear system: A @ u = b
            # where A[k, r] = V[j*p+k, r] * W[k*n+i, r]
            # and b = ones(p)
            A = np.zeros((p, r))
            for k in range(p):
                A[k, :] = V[j * p + k, :] * W[k * n + i, :]
            b = np.ones(p)
            try:
                sol, _, _, _ = linalg.lstsq(A, b, cond=None)
                U_new[i * m + j, :] = sol
            except:
                U_new[i * m + j, :] = U[i * m + j, :]

    # Solve for V
    for j in range(m):
        for k in range(p):
            # For fixed (j,k), we need:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
            # A[i, r] = U[i*m+j, r] * W[k*n+i, r]
            # b = ones(n)
            A = np.zeros((n, r))
            for i in range(n):
                A[i, :] = U_new[i * m + j, :] * W[k * n + i, :]
            b = np.ones(n)
            try:
                sol, _, _, _ = linalg.lstsq(A, b, cond=None)
                V_new[j * p + k, :] = sol
            except:
                V_new[j * p + k, :] = V[j * p + k, :]

    # Solve for W
    for k in range(p):
        for i in range(n):
            # For fixed (k,i), we need:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
            # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
            # b = ones(m)
            A = np.zeros((m, r))
            for j in range(m):
                A[j, :] = U_new[i * m + j, :] * V_new[j * p + k, :]
            b = np.ones(m)
            try:
                sol, _, _, _ = linalg.lstsq(A, b, cond=None)
                W_new[k * n + i, :] = sol
            except:
                W_new[k * n + i, :] = W[k * n + i, :]

    return U_new, V_new, W_new

def _try_reduce_rank(n, m, p, target_rank, max_iter=200, seed=42):
    """Try to find a decomposition of rank target_rank using ALS.
    Returns (U, V, W) if successful, else None.
    """
    if target_rank >= n * m * p:
        return _schoolbook()
    
    np.random.seed(seed)
    
    # Initialize with random factors
    U = np.random.randn(n * m, target_rank) / np.sqrt(target_rank)
    V = np.random.randn(m * p, target_rank) / np.sqrt(target_rank)
    W = np.random.randn(p * n, target_rank) / np.sqrt(target_rank)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = float('inf')
    
    prev_err = float('inf')
    stall_count = 0
    
    for it in range(max_iter):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        
        U, V, W = _als_iteration(U, V, W, n, m, p)
        err = _tensor_error(U, V, W, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            # Found an exact solution
            return best_U, best_V, best_W
        
        # Detect stall
        if err > prev_err - 1e-12:
            stall_count += 1
        else:
            stall_count = 0
        prev_err = err
        
        if stall_count > 50:
            break
    
    if best_err < 1e-8:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_U, schoolbook_V, schoolbook_W = _schoolbook()
    schoolbook_rank = n * m * p
    
    best_U, best_V, best_W = schoolbook_U, schoolbook_V, schoolbook_W
    best_rank = schoolbook_rank
    
    # Try to reduce rank from lower bound up to schoolbook rank
    # Lower bound for 3x4x4 matrix multiplication: max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # But actually the lower bound is max(n*m, m*p, p*n) for the trivial bound
    # More refined: the rank must be at least max(n*m, m*p, p*n)
    # Let's try ranks from 16 up to schoolbook_rank - 1
    
    lower_bound = max(n * m, m * p, p * n)
    
    for target_rank in range(lower_bound, schoolbook_rank):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        
        result = _try_reduce_rank(n, m, p, target_rank, max_iter=100, seed=target_rank)
        if result is not None:
            U, V, W = result
            # Verify
            err = _tensor_error(U, V, W, n, m, p)
            if err < 1e-10:
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                # Continue trying to go even lower? No, we're going up.
                # Actually we want the minimum, so break here since we found the lowest
                break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U.tolist()]
    V_list = [[float(x) for x in row] for row in best_V.tolist()]
    W_list = [[float(x) for x in row] for row in best_W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
