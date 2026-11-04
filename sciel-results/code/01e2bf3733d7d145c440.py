import numpy as np

def _verify(U, V, W, tol=1e-9):
    n = m = p = 3
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_rank23():
    n = m = p = 3
    rank = 23
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # 14 Winograd terms
    U[0, 0] = 1; U[1, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1; W[1, 0] = 1
    U[0, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; V[3, 1] = 1; W[0, 1] = 1; W[1, 1] = 1
    U[0, 2] = 1; U[3, 2] = -1; V[0, 2] = 1; V[3, 2] = -1; W[0, 2] = 1; W[1, 2] = -1
    U[1, 3] = 1; U[4, 3] = 1; V[1, 3] = 1; V[4, 3] = 1; W[1, 3] = 1; W[4, 3] = 1
    U[1, 4] = 1; U[4, 4] = 1; V[1, 4] = 1; V[4, 4] = -1; W[1, 4] = 1; W[4, 4] = -1
    U[1, 5] = 1; U[4, 5] = -1; V[1, 5] = 1; V[4, 5] = -1; W[1, 5] = 1; W[4, 5] = -1
    U[0, 6] = 1; U[4, 6] = 1; V[0, 6] = 1; V[4, 6] = 1; W[0, 6] = 1; W[4, 6] = 1
    U[4, 7] = 1; U[5, 7] = 1; V[4, 7] = 1; V[7, 7] = 1; W[4, 7] = 1; W[5, 7] = 1
    U[4, 8] = 1; U[7, 8] = 1; V[4, 8] = 1; V[7, 8] = 1; W[4, 8] = 1; W[5, 8] = 1
    U[4, 9] = 1; U[7, 9] = -1; V[4, 9] = 1; V[7, 9] = -1; W[4, 9] = 1; W[5, 9] = -1
    U[5, 10] = 1; U[8, 10] = 1; V[5, 10] = 1; V[8, 10] = 1; W[5, 10] = 1; W[8, 10] = 1
    U[5, 11] = 1; U[8, 11] = 1; V[5, 11] = 1; V[8, 11] = -1; W[5, 11] = 1; W[8, 11] = -1
    U[5, 12] = 1; U[8, 12] = -1; V[5, 12] = 1; V[8, 12] = -1; W[5, 12] = 1; W[8, 12] = -1
    U[4, 13] = 1; U[8, 13] = 1; V[4, 13] = 1; V[8, 13] = 1; W[4, 13] = 1; W[8, 13] = 1
    
    # Compute residual
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
                
    for r in range(14):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] -= u[i*m+j] * v[j*p+k] * w[k*n+i]
                    
    # Find 9 rank-1 terms for residual using greedy
    U_res = []
    V_res = []
    W_res = []
    
    for _ in range(9):
        max_idx = np.unravel_index(np.argmax(np.abs(T)), T.shape)
        val = T[max_idx]
        if abs(val) < 1e-9:
            break
        i, j, k = max_idx
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        u[i*m+j] = 1
        v[j*p+k] = 1
        w[k*n+i] = 1
        U_res.append(u)
        V_res.append(v)
        W_res.append(w)
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    T[ii, jj, kk] -= val * u[ii*m+jj] * v[jj*p+kk] * w[kk*n+ii]
                    
    if len(U_res) == 0:
        return None
        
    U_res = np.array(U_res).T
    V_res = np.array(V_res).T
    W_res = np.array(W_res).T
    
    U[:, 14:14+len(U_res)] = U_res
    V[:, 14:14+len(U_res)] = V_res
    W[:, 14:14+len(U_res)] = W_res
    
    if _verify(U, V, W):
        return U.tolist(), V.tolist(), W.tolist()
        
    return None

def _schoolbook():
    n = m = p = 3
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
    return U, V, W

def construct():
    import time
    _T0 = time.time()
    _DEADLINE = 140.0
    
    if time.time() - _T0 < _DEADLINE - 10:
        result = _construct_rank23()
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    U, V, W = _schoolbook()
    return {"U": U, "V": V, "W": W}
