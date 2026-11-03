"""Matrix multiplication 2x3 x 3x6, rank-minimal exact tensor decomposition."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 6]

def _find_2x3x3_rank15():
    """
    Attempts to find a rank-15 exact decomposition for 2x3x3 matrix multiplication.
    Returns U (6x15), V (9x15), W (6x15) if found, else None.
    """
    n, m, p = 2, 3, 3
    rank = 15
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def residuals(params):
        U = params[0:90].reshape(n*m, rank)
        V = params[90:225].reshape(m*p, rank)
        W = params[225:315].reshape(p*n, rank)
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_hat - T_target).flatten()

    start_time = time.time()
    timeout = 100.0 # 100 seconds for this sub-problem
    
    # Try multiple random initializations
    for seed in range(50):
        if time.time() - start_time > timeout:
            break
            
        np.random.seed(seed)
        # Initialize with small random values
        x0 = np.random.randn(315) * 0.5
        
        try:
            # Use 'trf' for better robustness
            res = least_squares(residuals, x0, method='trf', max_nfev=2000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            
            # Check if solution is good
            if res.cost < 1e-8:
                U = res.x[0:90].reshape(n*m, rank)
                V = res.x[90:225].reshape(m*p, rank)
                W = res.x[225:315].reshape(p*n, rank)
                
                # Round to nearest half-integer
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify exactly
                T_hat = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                if np.allclose(T_hat, T_target, atol=1e-9):
                    return U_int, V_int, W_int
        except Exception:
            continue
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find rank 30 solution via 2x3x3 block
    block = _find_2x3x3_rank15()
    if block is not None and time.time() - start_time < 115:
        U_block, V_block, W_block = block
        R = 30
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # First half (cols 0-2 of B, cols 0-2 of C)
        U[:, 0:15] = U_block
        V[0:9, 0:15] = V_block
        W[0:6, 0:15] = W_block
        
        # Second half (cols 3-5 of B, cols 3-5 of C)
        U[:, 15:30] = U_block
        V[9:18, 15:30] = V_block
        W[6:12, 15:30] = W_block
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to trivial rank 36
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
