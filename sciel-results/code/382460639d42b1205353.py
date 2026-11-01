"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank(r, n, m, p, max_iters=200, n_restarts=40, time_budget=8):
    deadline = _start_time + time_budget
    U_dim, V_dim, W_dim = n*m, m*p, p*n
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        np.random.seed(restart * 104729 + r)
        U = np.random.randn(U_dim, r)
        V = np.random.randn(V_dim, r)
        W = np.random.randn(W_dim, r)
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            for i in range(n):
                for j in range(m):
                    A = np.array([V[j*p+k]*W[k*n+i] for k in range(p)])
                    U[i*m+j] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
            
            for j in range(m):
                for k in range(p):
                    A = np.array([U[i*m+j]*W[k*n+i] for i in range(n)])
                    V[j*p+k] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
            
            for k in range(p):
                for i in range(n):
                    A = np.array([U[i*m+j]*V[j*p+k] for j in range(m)])
                    W[k*n+i] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
        
        U_l = [[float(x) for x in row] for row in U]
        V_l = [[float(x) for x in row] for row in V]
        W_l = [[float(x) for x in row] for row in W]
        if _verify(U_l, V_l, W_l, n, m, p, 1e-7):
            return (U_l, V_l, W_l)
    return None

def construct():
    n, m, p = DIMENSIONS
    for r in range(30, 61):
        if time.time() > _start_time + 140:
            break
        res = _try_rank(r, n, m, p, max_iters=150, n_restarts=30, time_budget=4)
        if res:
            U, V, W = res
            Uc = [[round(x*2)/2 for x in row] for row in U]
            Vc = [[round(x*2)/2 for x in row] for row in V]
            Wc = [[round(x*2)/2 for x in row] for row in W]
            if _verify(Uc, Vc, Wc, n, m, p, 1e-9):
                return {"U": Uc, "V": Vc, "W": Wc}
            return {"U": U, "V": V, "W": W}
    
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
