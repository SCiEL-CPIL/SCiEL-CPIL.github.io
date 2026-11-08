import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def check(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find lower rank decompositions
    for r in range(18, best_rank):
        if time.time() > deadline - 15:
            break
        
        found = False
        for seed in range(50):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed)
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Alternating least squares
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Solve for W
                for k in range(p):
                    for i in range(n):
                        A = np.array([U[i*m+j] * V[j*p+k] for j in range(m)])
                        b = np.ones(m)
                        W[k*n+i], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                
                # Solve for U
                for i in range(n):
                    for j in range(m):
                        A = np.array([V[j*p+k] * W[k*n+i] for k in range(p)])
                        b = np.ones(p)
                        U[i*m+j], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                
                # Solve for V
                for j in range(m):
                    for k in range(p):
                        A = np.array([U[i*m+j] * W[k*n+i] for i in range(n)])
                        b = np.ones(n)
                        V[j*p+k], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            if check(U, V, W):
                best_rank = r
                best_U, best_V, best_W = U, V, W
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
