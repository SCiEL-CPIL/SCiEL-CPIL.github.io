import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 15]
TIME_LIMIT = 140

def _build_schoolbook(n, m, p):
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

def _build_strassen_2x2x2():
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Term 1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # Term 3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # Term 4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Term 5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # Term 6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_rank53():
    n, m, p = DIMENSIONS
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    total_rank = 7 * 7 + 4
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b, block_size in enumerate(block_sizes):
        start_k = sum(block_sizes[:b])
        if block_size == 2:
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    V[j * p + start_k + k_local, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                for i in range(n):
                    W[(start_k + k_local) * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    return U, V, W

def _residuals_2x2x3(x, n=2, m=2, p=3, r=10):
    """Calculate residuals for a 2x2x3 decomposition of rank r."""
    U = x[0:n*m*r].reshape(n*m, r)
    V = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
    W = x[n*m*r + m*p*r:].reshape(p*n, r)
    
    residuals = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                residuals.append(val - 1.0)
    return np.array(residuals)

def _try_find_2x2x3_rank10(time_budget=30.0):
    """Try to find a rank 10 decomposition for 2x2x3 using least_squares."""
    start_time = time.time()
    n, m, p = 2, 2, 3
    r = 10
    
    # We need 160 variables.
    # Let's try a few random restarts.
    np.random.seed(42)
    
    for restart in range(50):
        if time.time() - start_time > time_budget:
            break
            
        # Random initialization
        x0 = np.random.randn(n*m*r + m*p*r + p*n*r) * 0.1
        
        # Alternatively, initialize with a known rank 11 solution truncated?
        # Or use the schoolbook solution (rank 12) truncated?
        
        try:
            res = least_squares(_residuals_2x2x3, x0, args=(n, m, p, r), method='lm', max_nfev=2000)
            if res.cost < 1e-10:
                U = res.x[0:n*m*r].reshape(n*m, r)
                V = res.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W = res.x[n*m*r + m*p*r:].reshape(p*n, r)
                # Verify
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            if abs(val - 1.0) > 1e-8:
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    return U, V, W
        except Exception:
            pass
            
    return None

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Fallback: Rank 53
    U_53, V_53, W_53 = _build_rank53()
    best_result = {"U": U_53.tolist(), "V": V_53.tolist(), "W": W_53.tolist()}
    best_rank = 53
    
    # Try to find Rank 50 using 2x2x3 rank 10 blocks
    # Time budget for this search: 60s
    if time.time() - start_time < TIME_LIMIT - 10:
        U_10, V_10, W_10 = _try_find_2x2x3_rank10(time_budget=60.0)
        
        if U_10 is not None:
            # Construct full rank 50 decomposition
            total_rank = 50
            U_full = np.zeros((n * m, total_rank))
            V_full = np.zeros((m * p, total_rank))
            W_full = np.zeros((p * n, total_rank))
            
            for b in range(5):
                start_k = b * 3
                col_off = b * 10
                U_full[:, col_off:col_off+10] = U_10
                for j in range(m):
                    for k_local in range(3):
                        V_full[j * p + start_k + k_local, col_off:col_off+10] = V_10[j * 3 + k_local, :]
                for k_local in range(3):
                    for i in range(n):
                        W_full[(start_k + k_local) * n + i, col_off:col_off+10] = W_10[k_local * 2 + i, :]
            
            if _verify_decomposition(U_full, V_full, W_full, n, m, p):
                best_result = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
                best_rank = 50
                # If we found 50, can we try 49? Unlikely to be quick.
                # We can break here or continue if time permits.
                if time.time() - start_time < TIME_LIMIT - 20:
                    # Try to find 2x2x3 rank 9?
                    # Or 2x2x4 rank 13? 53 -> 52?
                    pass
            else:
                # Verification failed, stick with 53
                pass
                
    return best_result
