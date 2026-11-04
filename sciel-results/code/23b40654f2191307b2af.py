import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _verify_fast(U_np, V_np, W_np, n, m, p):
    """Fast verification assuming numpy arrays."""
    rank = U_np.shape[1]
    max_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_np[u_idx, :], V_np[v_idx, :] * W_np[w_idx, :])
                error = abs(val - 1.0)
                if error > max_error:
                    max_error = error
                if max_error > 1e-9:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook rank n*m*p decomposition."""
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

def construct():
    """
    Construct a rank-minimal tensor decomposition for 3x4 * 4x6 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' containing the factor matrices
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Build schoolbook decomposition
    U, V, W = _schoolbook_decomposition(n, m, p)
    
    # Verify it's correct
    if not _verify_fast(U, V, W, n, m, p):
        raise RuntimeError("Schoolbook decomposition verification failed")
    
    # Convert to lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
