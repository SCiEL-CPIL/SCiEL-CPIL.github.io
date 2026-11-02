import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 3, 6]

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1
    return T

def _verify_decomposition(U, V, W, T):
    tensor = np.einsum('ac,bc,dc->abd', U, V, W)
    if not np.allclose(tensor, T, atol=1e-9):
        return False
    for mat in [U, V, W]:
        for val in mat:
            if abs(2 * val - round(2 * val)) > 1e-6:
                return False
    return True

def _snap_to_half_integers(mat):
    return np.round(mat * 2) / 2

def _search_rank_r(r, T, time_limit=100):
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    
    def objective(x):
        U = x[:nm * r].reshape(nm, r)
        V = x[nm * r: nm * r + mp * r].reshape(mp, r)
        W = x[nm * r + mp * r:].reshape(pn, r)
        
        tensor = np.einsum('ac,bc,dc->abd', U, V, W)
        error = tensor - T
        return error.flatten()
    
    best_error = np.inf
    best_x = None
    
    # Try multiple random initializations
    for _ in range(100):
        if time.time() - start_time > time_limit:
            break
        
        x0 = np.random.randn(nm * r + mp * r + pn * r)
        x0 /= (np.linalg.norm(x0) * np.sqrt(r))
        
        try:
            res = opt.least_squares(objective, x0, method='trf', max_nfev=2000)
        except Exception:
            continue
        
        if res.cost < best_error:
            best_error = res.cost
            best_x = res.x
            
            if best_error < 1e-14:
                break
    
    if best_x is None:
        return None
    
    U = _snap_to_half_integers(best_x[:nm * r].reshape(nm, r))
    V = _snap_to_half_integers(best_x[nm * r: nm * r + mp * r].reshape(mp, r))
    W = _snap_to_half_integers(best_x[nm * r + mp * r:].reshape(pn, r))
    
    if _verify_decomposition(U, V, W, T):
        return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    T = _build_tensor()
    
    # Try rank 32
    result = _search_rank_r(32, T, time_limit=100)
    if result is not None:
        return result
        
    # Fallback to schoolbook rank 36
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
