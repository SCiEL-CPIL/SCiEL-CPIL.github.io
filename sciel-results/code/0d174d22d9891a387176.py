import numpy as np
from fractions import Fraction
import time
import sys

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The target tensor T[i,j,k] = 1 for all i,j,k
    # Check if sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Schoolbook fallback
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1

    deadline = time.time() + 110
    
    # Try rank 28
    target_rank = 28
    # Random search
    for attempt in range(1000):
        if time.time() > deadline:
            break
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # Quick check
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-6:
                        ok = False
                        break
                if not ok: break
            if not ok: break
        
        if ok:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U_fb, "V": V_fb, "W": W_fb}
