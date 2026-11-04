import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _schoolbook():
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

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    # Target tensor T[i, j, j', k] = delta_{j, j'}
    # We check sum_r U[i,j,r] V[j',k,r] W[k,i,r]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += Ua[i*m+j, c] * Va[jp*p+k, c] * Wa[k*n+i, c]
                    target = 1.0 if j == jp else 0.0
                    if abs(val - target) > tol:
                        return False
    return True

def _find_rank_23(deadline):
    """
    Attempts to find a rank-23 decomposition using numerical optimization.
    """
    n, m, p = DIMENSIONS
    rank = 23
    
    # Target tensor T[i, j, j', k] = delta_{j, j'}
    # Flattened indices for error calculation
    # We define the error as sum over all (i, j, jp, k) of (sum_r ... - delta)^2
    
    def error(x):
        # x is flattened U, V, W
        # U: 9 * 23, V: 9 * 23, W: 9 * 23
        size = 9 * 23
        U = x[0:size].reshape(9, 23)
        V = x[size:2*size].reshape(9, 23)
        W = x[2*size:3*size].reshape(9, 23)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    target = 1.0 if j == jp else 0.0
                    for k in range(p):
                        # Sum over r
                        s = np.sum(U[i*m+j, :] * V[jp*p+k, :] * W[k*n+i, :])
                        err += (s - target) ** 2
        return err

    # We try a few random restarts
    np.random.seed(42)
    best_err = float('inf')
    best_x = None
    
    # Time check
    start = time.time()
    max_restarts = 5
    
    for r in range(max_restarts):
        if time.time() > deadline - 10: # Leave 10s for verification and fallback
            break
            
        # Initialize with small random values
        x0 = np.random.randn(3 * 9 * 23) * 0.1
        
        # Use Nelder-Mead as it doesn't require gradients (which are expensive to compute analytically here)
        # Or use BFGS with numerical gradients? Nelder-Mead is slower per step but robust.
        # Given 621 variables, Nelder-Mead might be very slow.
        # Let's try a few steps of gradient descent with numerical gradient?
        
        # Actually, let's use a simpler heuristic:
        # Start with schoolbook rank 27, and try to zero out 4 rows?
        # No, rank 23 is structurally different.
        
        # Let's stick to a short run of minimize.
        try:
            res = minimize(error, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x
        except Exception:
            pass

    if best_x is not None and best_err < 1e-3:
        # Round to nearest 0.5
        U, V, W = best_x[:9*23].reshape(9, 23), best_x[9*23:18*23].reshape(9, 23), best_x[18*23:27*23].reshape(9, 23)
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _verify(U_list, V_list, W_list, tol=1e-6):
            return U_list, V_list, W_list

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Try to find rank 23
    if time.time() < deadline - 5:
        res = _find_rank_23(deadline)
        if res:
            return {"U": res[0], "V": res[1], "W": res[2]}
            
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
