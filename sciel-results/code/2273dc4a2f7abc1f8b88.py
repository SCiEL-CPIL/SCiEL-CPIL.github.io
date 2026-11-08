"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    deadline = time.time() + 110  # safety margin
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    def check_decomposition(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    # Try to find lower rank by random search with linear solves
    for r in range(30, rank_sb):
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(300000 + r * 31 + attempt)
            strategy = attempt % 3
            
            if strategy == 0:
                W = np.random.randn(pn, r)
                V = np.random.randn(mp, r)
                U = np.zeros((nm, r))
                ok = True
                for i in range(n):
                    for j in range(m):
                        A = np.array([[W[k*n+i, c] * V[j*p+k, c] for c in range(r)] for k in range(p)])
                        b = np.ones(p)
                        sol, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        if not np.allclose(A @ sol, b, atol=1e-6):
                            ok = False
                            break
                        U[i*m+j, :] = sol
                    if not ok:
                        break
                if ok and check_decomposition(U, V, W):
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    found = True
                    break
                    
            elif strategy == 1:
                U = np.random.randn(nm, r)
                W = np.random.randn(pn, r)
                V = np.zeros((mp, r))
                ok = True
                for j in range(m):
                    for k in range(p):
                        A = np.array([[W[k*n+i, c] * U[i*m+j, c] for c in range(r)] for i in range(n)])
                        b = np.ones(n)
                        sol, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        if not np.allclose(A @ sol, b, atol=1e-6):
                            ok = False
                            break
                        V[j*p+k, :] = sol
                    if not ok:
                        break
                if ok and check_decomposition(U, V, W):
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    found = True
                    break
                    
            else:
                U = np.random.randn(nm, r)
                V = np.random.randn(mp, r)
                W = np.zeros((pn, r))
                ok = True
                for k in range(p):
                    for i in range(n):
                        A = np.array([[U[i*m+j, c] * V[j*p+k, c] for c in range(r)] for j in range(m)])
                        b = np.ones(m)
                        sol, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        if not np.allclose(A @ sol, b, atol=1e-6):
                            ok = False
                            break
                        W[k*n+i, :] = sol
                    if not ok:
                        break
                if ok and check_decomposition(U, V, W):
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    found = True
                    break
        
        if found:
            break
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
