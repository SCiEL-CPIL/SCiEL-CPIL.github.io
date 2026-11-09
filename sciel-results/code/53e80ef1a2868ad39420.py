"""Matrix multiplication 3,3,5 - rank-minimal tensor decomposition"""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _decompose_to_tensor(U, V, W):
    n, m, p = DIMENSIONS
    r = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for c in range(r):
        T += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    return T

def _als_step(U, V, W, T, max_iter=100, tol=1e-12):
    n, m, p = DIMENSIONS
    r = U.shape[1]
    for it in range(max_iter):
        VtV = V.T @ V
        WtW = W.T @ W
        M_U = VtV * WtW
        rhs_U = np.einsum('abc,br,cr->ar', T, V, W)
        try:
            U_new = rhs_U @ np.linalg.inv(M_U)
        except np.linalg.LinAlgError:
            U_new = rhs_U @ np.linalg.pinv(M_U)
        
        UtU = U_new.T @ U_new
        WtW = W.T @ W
        M_V = UtU * WtW
        rhs_V = np.einsum('abc,ar,cr->br', T, U_new, W)
        try:
            V_new = rhs_V @ np.linalg.inv(M_V)
        except np.linalg.LinAlgError:
            V_new = rhs_V @ np.linalg.pinv(M_V)
        
        UtU = U_new.T @ U_new
        VtV = V_new.T @ V_new
        M_W = UtU * VtV
        rhs_W = np.einsum('abc,ar,br->cr', T, U_new, V_new)
        try:
            W_new = rhs_W @ np.linalg.inv(M_W)
        except np.linalg.LinAlgError:
            W_new = rhs_W @ np.linalg.pinv(M_W)
        
        U, V, W = U_new, V_new, W_new
        T_hat = _decompose_to_tensor(U, V, W)
        err = np.linalg.norm(T - T_hat) / (np.linalg.norm(T) + 1e-15)
        if err < tol:
            break
    
    return U, V, W, err

def _try_rank(T, rank, n_restarts=5, time_limit=8):
    n, m, p = DIMENSIONS
    start_time = time.time()
    best = None
    best_err = np.inf
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        U, V, W, err = _als_step(U, V, W, T, max_iter=200, tol=1e-10)
        
        if err < best_err:
            best_err = err
            best = (U, V, W, err)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j][col] = 1
                V_school[j*p + k][col] = 1
                W_school[k*n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try lower ranks
    for rank in range(15, rank_school):
        if time.time() > deadline - 5:
            break
        result = _try_rank(T, rank, n_restarts=3, time_limit=8)
        if result:
            U, V, W, err = result
            T_hat = _decompose_to_tensor(U, V, W)
            max_diff = np.max(np.abs(T - T_hat))
            if max_diff < 1e-6:
                U_s = np.round(U * 2) / 2
                V_s = np.round(V * 2) / 2
                W_s = np.round(W * 2) / 2
                T_s = _decompose_to_tensor(U_s, V_s, W_s)
                if np.max(np.abs(T - T_s)) < 1e-8:
                    best_result = {
                        "U": [[float(x) for x in U_s[i]] for i in range(n*m)],
                        "V": [[float(x) for x in V_s[i]] for i in range(m*p)],
                        "W": [[float(x) for x in W_s[i]] for i in range(p*n)]
                    }
                    break
                elif max_diff < 1e-8:
                    best_result = {
                        "U": [[float(x) for x in U[i]] for i in range(n*m)],
                        "V": [[float(x) for x in V[i]] for i in range(m*p)],
                        "W": [[float(x) for x in W[i]] for i in range(p*n)]
                    }
                    break
    
    return best_result
# EVOLVE-BLOCK-END
