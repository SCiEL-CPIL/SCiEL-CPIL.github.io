"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 54
    
    # Build the schoolbook decomposition as fallback
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    def to_half_integer(x):
        """Round to nearest 0.5"""
        return np.round(2.0 * x) / 2.0
    
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Check if decomposition is exact"""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def verify_strict(U, V, W, n, m, p):
        """Check if decomposition is exact with strict integer/half-integer check"""
        # Check all entries are multiples of 0.5
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return verify(U, V, W, n, m, p, tol=1e-12)
    
    def try_rank(r, seed, time_limit):
        """Try to find a rank-r decomposition using structured ALS"""
        start = time.time()
        np.random.seed(seed)
        
        # Structured initialization: start from a subset of schoolbook columns
        # or use a low-rank approximation of the schoolbook
        if r >= rank_schoolbook:
            # Just use schoolbook subset
            U = U_sb[:, :r].copy()
            V = V_sb[:, :r].copy()
            W = W_sb[:, :r].copy()
        else:
            # Use SVD-based low-rank init on the "flattened" schoolbook
            # Or random with scaling
            scale = np.sqrt(rank_schoolbook / r)
            U = np.random.randn(n * m, r) * scale
            V = np.random.randn(m * p, r) * scale
            W = np.random.randn(p * n, r) / scale  # balance
        
        # ALS iterations with polish
        for iteration in range(80):
            if time.time() - start > time_limit:
                break
            
            # Update U: for each (i,j), solve A * U[i*m+j,:] = ones(p)
            # where A[k, r] = V[j*p+k, r] * W[k*n+i, r]
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve A * V[j*p+k,:] = ones(n)
            # where A[i, r] = U[i*m+j, r] * W[k*n+i, r]
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), solve A * W[k*n+i,:] = ones(m)
            # where A[j, r] = U[i*m+j, r] * V[j*p+k, r]
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                    except:
                        pass
        
        # Polish: more iterations with tighter tolerance
        for iteration in range(40):
            if time.time() - start > time_limit:
                break
            
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=1e-14)[0]
                    except:
                        pass
            
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A, np.ones(n), rcond=1e-14)[0]
                    except:
                        pass
            
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A, np.ones(m), rcond=1e-14)[0]
                    except:
                        pass
        
        # Round to half-integers and verify
        U_h = to_half_integer(U)
        V_h = to_half_integer(V)
        W_h = to_half_integer(W)
        
        if verify_strict(U_h, V_h, W_h, n, m, p):
            return (U_h, V_h, W_h)
        
        # Also try without rounding in case the solution is already clean
        if verify(U, V, W, n, m, p, tol=1e-10):
            # Check if close to half-integers
            U_close = to_half_integer(U)
            V_close = to_half_integer(V)
            W_close = to_half_integer(W)
            if verify_strict(U_close, V_close, W_close, n, m, p):
                return (U_close, V_close, W_close)
        
        return None
    
    # Try ranks from 53 down to 36
    # Use more seeds for lower ranks
    for r in range(53, 35, -1):
        if time.time() > deadline - 15:
            break
        max_seeds = 20 if r <= 45 else 10
        for seed in range(max_seeds):
            if time.time() > deadline - 10:
                break
            # Allocate time based on remaining budget
            remaining = deadline - time.time()
            time_per_try = min(3.0, remaining / (max_seeds - seed + 1))
            result = try_rank(r, seed, time_limit=time_per_try)
            if result is not None:
                U_h, V_h, W_h = result
                return {
                    "U": U_h.tolist(),
                    "V": V_h.tolist(),
                    "W": W_h.tolist()
                }
    
    # Fall back to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END
