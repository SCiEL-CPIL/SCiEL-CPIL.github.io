"""Matrix multiplication 4,4,6 rank-minimal decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    def verify(U, V, W):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def to_list(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def make_schoolbook():
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
        return U, V, W, rank
    
    U0, V0, W0, rank0 = make_schoolbook()
    
    random.seed(42)
    np.random.seed(42)
    
    best_rank = rank0
    best_U, best_V, best_W = U0, V0, W0
    
    # Try to find minimum rank
    # For a given rank r, choose random U, V, then solve for W
    for r in range(1, min(30, rank0 + 1)):
        if time.time() > deadline - 15:
            break
        found = False
        for attempt in range(50):
            if time.time() > deadline - 15:
                break
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            
            W = np.zeros((p*n, r))
            feasible = True
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j] = U[i*m+j] * V[j*p+k]
                    try:
                        b, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        if np.linalg.norm(A @ b - np.ones(m)) > 1e-9:
                            feasible = False
                            break
                        W[k*n+i] = b
                    except:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if feasible and verify(U, V, W):
                found = True
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break
        
        if found:
            break
    
    return to_list(best_U, best_V, best_W)
# EVOLVE-BLOCK-END
