import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

def _verify_factors(U, V, W, n, m, p, tol=1e-6):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                expected = 1
                actual = np.dot(U_arr[A], V_arr[B] * W_arr[C])
                if abs(actual - expected) > tol:
                    return False, (A, B, C, expected, actual)
    return True, None

def _build_schoolbook(n, m, p):
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
    return U, V, W

def _solve_for_rank(n, m, p, R, max_iter=1000, tol=1e-9):
    """
    Attempt to find a decomposition of rank R using least squares.
    """
    n_vars = R * (n * m + m * p + p * n)
    
    def residual(x):
        U_flat = x[:R * n * m]
        V_flat = x[R * n * m : R * (n * m + m * p)]
        W_flat = x[R * (n * m + m * p) :]
        
        U = U_flat.reshape(R, n * m).T # (n*m, R)
        V = V_flat.reshape(R, m * p).T # (m*p, R)
        W = W_flat.reshape(R, p * n).T # (p*n, R)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A = i * m + j
                    B = j * p + k
                    C = k * n + i
                    # sum_r U[A, r] V[B, r] W[C, r]
                    val = np.dot(U[A], V[B] * W[C])
                    res.append(val - 1.0)
        return np.array(res)
    
    # Try a few random starts
    for attempt in range(10):
        x0 = np.random.randn(n_vars) * 0.1
        # Use scipy least squares
        try:
            from scipy.optimize import least_squares
            res = least_squares(residual, x0, method='lm', xtol=1e-12, ftol=1e-12, max_nfev=max_iter)
            if res.cost < 1e-10:
                # Reconstruct
                U_flat = res.x[:R * n * m]
                V_flat = res.x[R * n * m : R * (n * m + m * p)]
                W_flat = res.x[R * (n * m + m * p) :]
                
                U = U_flat.reshape(R, n * m).T.tolist()
                V = V_flat.reshape(R, m * p).T.tolist()
                W = W_flat.reshape(R, p * n).T.tolist()
                
                # Verify
                ok, _ = _verify_factors(U, V, W, n, m, p)
                if ok:
                    return U, V, W, R
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120 - 5
    
    # Try to find a lower rank decomposition
    # Start from 17 down to 11
    for R in range(17, 10, -1):
        if time.time() - start_time > deadline:
            break
            
        result = _solve_for_rank(n, m, p, R)
        if result is not None:
            U, V, W, rank = result
            return {"U": U, "V": V, "W": W}
            
    # Fall back to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
