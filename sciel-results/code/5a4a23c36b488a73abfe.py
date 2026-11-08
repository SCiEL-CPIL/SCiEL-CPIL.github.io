import numpy as np
import time

def construct():
    n, m, p = 2, 3, 7
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to find a low-rank decomposition
    for rank in range(21, 43):
        if time.time() > deadline - 10:
            break
        result = find_decomposition(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook(n, m, p)

def schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def find_decomposition(n, m, p, rank, deadline):
    for seed in range(30):
        if time.time() > deadline - 5:
            break
        np.random.seed(seed)
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        for it in range(3000):
            if time.time() > deadline - 2:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.array([V[j * p + k, :] * W[k * n + i, :] for k in range(p)])
                    U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])
                    V[j * p + k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                    W[k * n + i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
            
            # Check convergence
            err = compute_error(U, V, W, n, m, p)
            if err < 1e-10:
                result = try_round(U, V, W, n, m, p)
                if result is not None:
                    return result
    
    return None

def compute_error(U, V, W, n, m, p):
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j, :]
            for k in range(p):
                val = np.dot(u_row, V[j * p + k, :] * W[k * n + i, :])
                max_err = max(max_err, abs(val - 1.0))
    return max_err

def try_round(U, V, W, n, m, p):
    # Try rounding to nearest half-integer
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    err = compute_error(U_r, V_r, W_r, n, m, p)
    if err < 1e-8:
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Try rounding to nearest integer
    U_r = np.round(U)
    V_r = np.round(V)
    W_r = np.round(W)
    err = compute_error(U_r, V_r, W_r, n, m, p)
    if err < 1e-8:
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return None
