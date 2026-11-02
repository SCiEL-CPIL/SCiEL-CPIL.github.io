import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 3, 3
    deadline = time.time() + 140
    
    # Precompute the identity tensor
    T_id = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_id[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try ranks from 17 down to 16
    for R in range(17, 15, -1):
        if time.time() > deadline:
            break
        result = _try_find_decomposition(n, m, p, R, T_id, deadline)
        if result is not None:
            U, V, W = result
            return {
                "U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]
            }
    
    # Fallback to schoolbook (rank 18)
    U, V, W = _schoolbook(n, m, p)
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

def _residuals(params, n, m, p, R, T_id):
    U = params[0:n*m*R].reshape(n*m, R)
    V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
    W = params[n*m*R + m*p*R:].reshape(p*n, R)
    
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    return (T - T_id).ravel()

def _try_find_decomposition(n, m, p, R, T_id, deadline):
    total_vars = n*m*R + m*p*R + p*n*R
    np.random.seed(42)
    
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        params = np.random.randn(total_vars) * 0.5
        
        try:
            result = least_squares(
                _residuals, params, 
                args=(n, m, p, R, T_id),
                method='trf',
                max_nfev=100,
                xtol=1e-14, ftol=1e-14, gtol=1e-14
            )
        except:
            continue
        
        if result.cost < 1e-28:
            U = result.x[0:n*m*R].reshape(n*m, R)
            V = result.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
            W = result.x[n*m*R + m*p*R:].reshape(p*n, R)
            if _verify(U, V, W, n, m, p, T_id, tol=1e-10):
                return U, V, W
    
    return None

def _verify(U, V, W, n, m, p, T_id, tol=1e-9):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    if not np.allclose(T, T_id, atol=tol):
        return False
    return True
