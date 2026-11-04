import numpy as np
import time
import itertools
from scipy.optimize import least_squares

DIMENSIONS = [2, 5, 5]

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    # Reshape to tensor form for efficient computation?
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can compute this efficiently using einsum or loops.
    # Given small sizes, loops are fine, but einsum is cleaner.
    # U is (n*m, R), V is (m*p, R), W is (p*n, R)
    # We want T[i, j, k]
    # U_idx = i*m + j
    # V_idx = j*p + k
    # W_idx = k*n + i
    
    # Let's just use the loop for clarity and safety with indices.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _objective(params, n, m, p, R):
    """
    params: flattened array of U, V, W
    Returns: flattened error vector
    """
    size_U = n * m * R
    size_V = m * p * R
    size_W = p * n * R
    
    U_flat = params[:size_U]
    V_flat = params[size_U:size_U+size_V]
    W_flat = params[size_U+size_V:]
    
    U = U_flat.reshape(n * m, R)
    V = V_flat.reshape(m * p, R)
    W = W_flat.reshape(p * n, R)
    
    errors = np.zeros(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                errors[idx] = val - 1.0
                idx += 1
    return errors

def _try_rank(n, m, p, R, max_time=10, max_restarts=5):
    """
    Try to find a rank R decomposition using least squares.
    """
    start_time = time.time()
    size_U = n * m * R
    size_V = m * p * R
    size_W = p * n * R
    total_size = size_U + size_V + size_W
    
    # Initialize with random values
    # To help convergence, we can start with small random values
    for _ in range(max_restarts):
        if time.time() - start_time > max_time:
            break
            
        x0 = np.random.normal(0, 0.1, total_size)
        
        try:
            res = least_squares(_objective, x0, args=(n, m, p, R), method='lm', max_nfev=1000)
            if res.success or res.cost < 1e-8:
                # Check validity
                size_U = n * m * R
                size_V = m * p * R
                size_W = p * n * R
                U = res.x[:size_U].reshape(n * m, R)
                V = res.x[size_U:size_U+size_V].reshape(m * p, R)
                W = res.x[size_U+size_V:].reshape(p * n, R)
                
                if _verify(U, V, W, n, m, p, tol=1e-5):
                    return U, V, W
        except Exception:
            continue
            
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    start_time = time.time()
    deadline = 140  # seconds, keep 10s margin
    
    # Fallback
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    
    # Try ranks from 20 to 50
    # We know rank(2,2,2)=7, rank(2,3,3)=11.
    # For 2,5,5, rank is likely around 20-25.
    # Let's try 20, 21, ..., 50.
    for R in range(20, rank_school + 1):
        if time.time() - start_time > deadline:
            break
        
        # Allocate time for this rank
        time_left = deadline - (time.time() - start_time)
        if time_left < 1:
            break
            
        # Try to find solution for this rank
        U, V, W = _try_rank(n, m, p, R, max_time=time_left, max_restarts=3)
        
        if U is not None:
            best_U, best_V, best_W, best_rank = U, V, W, R
            break # Found minimal rank, stop
            
    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
