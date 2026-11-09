"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _schoolbook_decomp(n, m, p):
    """Return schoolbook decomposition for n x m * m x p -> n x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition is exact (within tolerance)."""
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _round_to_half(x):
    """Round to nearest half-integer."""
    return np.round(x * 2) / 2.0

def _als_step(U, V, W, n, m, p, R):
    """One ALS iteration with vectorized updates."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Reshape for clarity: U is (n*m, R), V is (m*p, R), W is (p*n, R)
    
    # Update W: for each (i, k), solve A @ W_new = ones(m)
    # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
    W_new = W.copy()
    for i in range(n):
        for k in range(p):
            A = np.zeros((m, R))
            for j in range(m):
                A[j, :] = U[i*m+j, :] * V[j*p+k, :]
            try:
                W_new[k*n+i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
            except:
                pass
    W = W_new
    
    # Update U: for each (i, j), solve A @ U_new = ones(p)
    # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
    U_new = U.copy()
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, R))
            for k in range(p):
                A[k, :] = V[j*p+k, :] * W[k*n+i, :]
            try:
                U_new[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
            except:
                pass
    U = U_new
    
    # Update V: for each (j, k), solve A @ V_new = ones(n)
    # A[i, r] = U[i*m+j, r] * W[k*n+i, r]
    V_new = V.copy()
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, R))
            for i in range(n):
                A[i, :] = U[i*m+j, :] * W[k*n+i, :]
            try:
                V_new[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
            except:
                pass
    V = V_new
    
    return U, V, W

def _try_rank(n, m, p, R, num_trials=20, iters_per_trial=50, timeout=10):
    """Try to find a rank-R decomposition. Return dict or None."""
    start = time.time()
    
    for trial in range(num_trials):
        if time.time() - start > timeout:
            break
        
        # Random initialization
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        for it in range(iters_per_trial):
            U, V, W = _als_step(U, V, W, n, m, p, R)
            
            # Check convergence
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        error += (val - 1.0)**2
            
            if error < 1e-10:
                # Try to round to half-integers
                U_r = _round_to_half(U)
                V_r = _round_to_half(V)
                W_r = _round_to_half(W)
                
                if _verify(U_r, V_r, W_r, n, m, p, tol=1e-6):
                    return {
                        'U': U_r.tolist(),
                        'V': V_r.tolist(),
                        'W': W_r.tolist()
                    }
        
        # Also check without rounding
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    error += (val - 1.0)**2
        
        if error < 1e-10:
            U_r = _round_to_half(U)
            V_r = _round_to_half(V)
            W_r = _round_to_half(W)
            
            if _verify(U_r, V_r, W_r, n, m, p, tol=1e-6):
                return {
                    'U': U_r.tolist(),
                    'V': V_r.tolist(),
                    'W': W_r.tolist()
                }
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    schoolbook_rank = n * m * p  # 96
    
    # Try ranks from 50 to 95 in descending order of likelihood
    # Start with lower ranks first as they're more interesting
    for target_rank in range(50, schoolbook_rank):
        if time.time() > deadline:
            break
        
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
        
        # Allocate time based on remaining budget
        timeout = min(5, remaining_time * 0.5)
        num_trials = max(5, int(timeout * 2))
        
        result = _try_rank(n, m, p, target_rank, 
                          num_trials=num_trials, 
                          iters_per_trial=30, 
                          timeout=timeout)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook_decomp(n, m, p)
# EVOLVE-BLOCK-END
