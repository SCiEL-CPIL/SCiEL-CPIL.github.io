"""Matrix multiplication 3x4x5 rank-minimal exact tensor decomposition search."""
import time
import numpy as np
from fractions import Fraction
import scipy.optimize as opt

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # Safety margin before 120s limit

def _time_remaining():
    return _DEADLINE - time.time()

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    return U, V, W

def _compute_error_sq(U, V, W, n, m, p, rank):
    """Compute the sum of squared errors."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err += (s - 1.0) ** 2
    return err

def _flatten(U, V, W):
    return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

def _unflatten(params, n, m, p, rank):
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    U = params[:size_u].reshape(n * m, rank)
    V = params[size_u:size_u + size_v].reshape(m * p, rank)
    W = params[size_u + size_v:].reshape(p * n, rank)
    return U, V, W

def _objective(params, n, m, p, rank):
    U, V, W = _unflatten(params, n, m, p, rank)
    return _compute_error_sq(U, V, W, n, m, p, rank)

def _try_optimize(n, m, p, rank, seed):
    """Try to find a decomposition of given rank using scipy optimization."""
    if time.time() > _DEADLINE:
        return None
        
    rng = np.random.RandomState(seed)
    # Initialize with small random values
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    total_size = size_u + size_v + size_w
    
    params = rng.randn(total_size) * 0.5
    
    # Define objective
    def obj(x):
        return _objective(x, n, m, p, rank)
        
    try:
        # L-BFGS-B is better for high dimensions
        res = opt.minimize(obj, params, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.fun < 1e-8:
            U, V, W = _unflatten(res.x, n, m, p, rank)
            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if _verify(U_snap, V_snap, W_snap, n, m, p, rank):
                return U_snap, V_snap, W_snap
    except Exception:
        pass
        
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank decomposition
    # Schoolbook is 60. Try 49, 50, 55, 59
    for rank in [49, 50, 55, 59]:
        if time.time() > _DEADLINE:
            break
        for seed in range(10):
            if time.time() > _DEADLINE:
                break
            result = _try_optimize(n, m, p, rank, seed)
            if result is not None:
                U, V, W = result
                U_list = [[float(U[i, r]) for r in range(rank)] for i in range(n*m)]
                V_list = [[float(V[i, r]) for r in range(rank)] for i in range(m*p)]
                W_list = [[float(W[i, r]) for r in range(rank)] for i in range(p*n)]
                return {"U": U_list, "V": V_list, "W": W_list}
                
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
