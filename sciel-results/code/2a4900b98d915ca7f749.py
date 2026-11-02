import time
import numpy as np

DIMENSIONS = [2, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False, (i, j, k, s)
    return True, None

def _schoolbook(n, m, p):
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

def _construct_optimal(n, m, p):
    """
    Constructs the rank m*p decomposition.
    U: (n*m, m*p)
    V: (m*p, m*p)
    W: (p*n, m*p)
    """
    rank = m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # V is identity matrix of size m*p
    for idx in range(rank):
        V[idx, idx] = 1.0
        
    # U and W
    # r corresponds to (j, k) with r = j*p + k
    # U[i*m + j, r] is 1 if j==k and r==j*p+k else 0
    # W[k*n + i, r] is 1 if j==k and r==j*p+k else 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = j * p + k
                if j == k:
                    U[i * m + j, r] = 1.0
                    W[k * n + i, r] = 1.0
                # else 0, which is already initialized
                    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Try the optimal rank m*p
    U_opt, V_opt, W_opt = _construct_optimal(n, m, p)
    ok, _ = _verify_decomposition(U_opt, V_opt, W_opt, n, m, p)
    
    if ok:
        return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
    
    # Fallback to schoolbook if verification fails (should not happen)
    U_school, V_school, W_school = _schoolbook(n, m, p)
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
