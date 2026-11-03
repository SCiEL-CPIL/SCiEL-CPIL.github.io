"""Matrix multiplication 3x5 x 5x6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W, rank

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                s = np.dot(U[a], np.multiply(V[b], W[c]))
                if abs(s - 1) > tol:
                    return False
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    if j2 == j:
                        continue
                    a = i*m + j
                    b = j2*p + k
                    for i2 in range(n):
                        c = k*n + i2
                        s = np.dot(U[a], np.multiply(V[b], W[c]))
                        if abs(s) > tol:
                            return False
    return True

def _block_decomp_356():
    n, m, p = 3, 5, 6
    m1, m2 = 2, 3
    total_rank = n*m1*p + n*m2*p  # 3*2*6 + 3*3*6 = 36+54=90
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W, total_rank

def _try_eliminate_columns(U, V, W, n, m, p, max_time=40):
    start = time.time()
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    rank = U.shape[1]
    
    for _ in range(100):
        if time.time() - start > max_time:
            break
        if rank <= 1:
            break
        
        removed = False
        for col in range(rank):
            U_bak = U[:, col].copy()
            V_bak = V[:, col].copy()
            W_bak = W[:, col].copy()
            
            U[:, col] = 0
            V[:, col] = 0
            W[:, col] = 0
            
            if _verify_decomp(U, V, W, n, m, p, tol=1e-8):
                cols = [c for c in range(rank) if c != col]
                U = U[:, cols]
                V = V[:, cols]
                W = W[:, cols]
                rank -= 1
                removed = True
                break
            
            U[:, col] = U_bak
            V[:, col] = V_bak
            W[:, col] = W_bak
        
        if not removed:
            break
    
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    U, V, W, rank = _schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    if time.time() < deadline - 30:
        U_e, V_e, W_e, rank_e = _try_eliminate_columns(U, V, W, n, m, p, max_time=50)
        if _verify_decomp(U_e, V_e, W_e, n, m, p):
            best_U, best_V, best_W, best_rank = U_e, V_e, W_e, rank_e
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
