import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                if abs(s - 1) > tol:
                    return False
    return True

def _round_to_half_int(arr, tol=1e-4):
    """Round array to nearest half-integer if close enough."""
    rounded = np.round(arr * 2) / 2
    if np.allclose(arr, rounded, atol=tol, rtol=0):
        return rounded
    return None

def _solve_rank_decomposition(rank=20, max_iters=2000, tol=1e-10):
    """
    Solve for a rank-r decomposition using least squares with good initialization.
    """
    n, m, p = 3, 3, 3
    np.random.seed(42)
    
    def flatten_params(x):
        U = x[:9*rank].reshape(9, rank)
        V = x[9*rank:18*rank].reshape(9, rank)
        W = x[18*rank:27*rank].reshape(9, rank)
        return U, V, W
    
    def residuals(x):
        U, V, W = flatten_params(x)
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    res.append(s - 1)
        return np.array(res)
    
    # Initialize with a structured pattern
    x0 = np.zeros(27 * rank)
    
    # Use schoolbook-like initialization for the first min(rank, 27) terms
    for c in range(min(rank, 27)):
        i, j, k = divmod(c, 9)
        j, k = divmod(j, 3)
        i, j = divmod(i, 1)
        # Actually, let's just use random initialization
        pass
    
    # Random initialization
    x0 = np.random.randn(27 * rank) * 0.1
    
    # Try multiple random initializations
    best_result = None
    best_err = float('inf')
    
    for trial in range(5):
        np.random.seed(42 + trial)
        x0 = np.random.randn(27 * rank) * 0.1
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=3000, xtol=1e-12, ftol=1e-12)
            err = np.max(np.abs(result.fun))
            if err < best_err:
                best_err = err
                best_result = result
        except Exception:
            continue
    
    if best_result is not None and best_err < 1e-8:
        U, V, W = flatten_params(best_result.x)
        
        # Try to round to half-integers
        U_r = _round_to_half_int(U)
        V_r = _round_to_half_int(V)
        W_r = _round_to_half_int(W)
        
        if U_r is not None and V_r is not None and W_r is not None:
            if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
        
        # Check if unrounded is valid
        if _verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-8):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Guaranteed rank-27 decomposition
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find rank 20
    if time.time() - start_time < timeout - 20:
        result = _solve_rank_decomposition(rank=20)
        if result is not None:
            U20, V20, W20 = result
            if _verify(U20, V20, W20):
                best_result = {"U": U20, "V": V20, "W": W20}
                best_rank = 20
    
    if time.time() - start_time > timeout:
        return best_result
    
    # Try rank 21 as fallback
    if best_rank > 21 and time.time() - start_time < timeout - 10:
        result = _solve_rank_decomposition(rank=21)
        if result is not None:
            U21, V21, W21 = result
            if _verify(U21, V21, W21):
                best_result = {"U": U21, "V": V21, "W": W21}
                best_rank = 21
    
    return best_result
# EVOLVE-BLOCK-END
