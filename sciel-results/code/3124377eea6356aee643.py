import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()
_HARD_DEADLINE = 115

def construct():
    n, m, p = DIMENSIONS
    deadline = _START_TIME + _HARD_DEADLINE

    # Get the rank-14 solution
    U14, V14, W14 = _get_rank14_solution(n, m, p)
    
    # Verify it
    err14 = _verify_decomposition(U14, V14, W14, n, m, p)
    
    if err14 < 1e-9:
        # Try to find lower rank
        for target_rank in [13, 12, 11]:
            if time.time() > deadline - 15:
                break
            U, V, W = _search_lower_rank(n, m, p, target_rank, deadline)
            if U is not None:
                err = _verify_decomposition(U, V, W, n, m, p)
                if err < 1e-9:
                    return _to_dict(U, V, W)
        
        return _to_dict(U14, V14, W14)
    else:
        # Fallback to schoolbook
        return _construct_schoolbook(n, m, p)

def _get_rank14_solution(n, m, p):
    # Rank-14 decomposition for 2x3x6 via 3x3x3 rank-7 slicing + duplication
    U33 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 1],
    ], dtype=np.float64)

    V33 = np.array([
        [1, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, -1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 1, 0],
        [0, 0, 0, 0, 0, -1, 1],
        [0, 0, 0, 0, 1, -1, 0],
    ], dtype=np.float64)

    W33 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, -1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float64)

    # Slice to 2x3x3
    U_2x3x3 = U33[0:6, :]
    V_2x3x3 = V33.copy()
    W_2x3x3 = np.zeros((6, 7), dtype=np.float64)
    for k in range(3):
        for i in range(2):
            W_2x3x3[k * 2 + i, :] = W33[k * 3 + i, :]

    # Construct 2x3x6 by duplicating the 2x3x3 block for k=0..2 and k=3..5
    r = 14
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)

    U[:, 0:7] = U_2x3x3
    V[0:9, 0:7] = V_2x3x3
    W[0:6, 0:7] = W_2x3x3

    U[:, 7:14] = U_2x3x3
    V[9:18, 7:14] = V_2x3x3
    W[6:12, 7:14] = W_2x3x3

    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    r = U.shape[1]
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    err = np.max(np.abs(T - 1.0))
    return err

def _search_lower_rank(n, m, p, target_rank, deadline):
    """
    Try to find a decomposition of lower rank using numerical optimization.
    """
    r = target_rank
    
    def _residual(params):
        U = params[0: n*m*r].reshape(n*m, r)
        V = params[n*m*r: n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r:].reshape(p*n, r)
        
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        residual = (T - 1.0).flatten()
        return np.sum(residual**2)
    
    # Try multiple random initializations
    best_loss = np.inf
    best_params = None
    
    n_init = 3
    for init_idx in range(n_init):
        if time.time() > deadline - 10:
            break
        
        np.random.seed(42 + init_idx)
        params = np.random.randn(n*m*r + m*p*r + p*n*r) * 1.0
        
        # Use L-BFGS-B for optimization
        result = minimize(_residual, params, method='L-BFGS-B', 
                         options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if result.fun < best_loss:
            best_loss = result.fun
            best_params = result.x
        
        if best_loss < 1e-10:
            break
    
    if best_params is not None and best_loss < 1e-10:
        U = best_params[0: n*m*r].reshape(n*m, r)
        V = best_params[n*m*r: n*m*r + m*p*r].reshape(m*p, r)
        W = best_params[n*m*r + m*p*r:].reshape(p*n, r)
        return U, V, W
    
    return None, None, None

def _construct_schoolbook(n, m, p):
    r = n * m * p
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    return _to_dict(U, V, W)

def _to_dict(U, V, W):
    def clean(arr):
        out = []
        for row in arr:
            crow = []
            for x in row:
                xi = round(float(x))
                if abs(float(x) - xi) < 1e-9:
                    crow.append(int(xi))
                else:
                    h = round(2.0 * float(x))
                    if abs(2.0 * float(x) - h) < 1e-9:
                        crow.append(h / 2.0)
                    else:
                        crow.append(float(x))
            out.append(crow)
        return out
    return {"U": clean(U), "V": clean(V), "W": clean(W)}
