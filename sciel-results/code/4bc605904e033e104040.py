import numpy as np
import time
from fractions import Fraction
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 3, 4
    
    def verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def error(U, V, W):
        rank = U.shape[1]
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (val - 1.0) ** 2
        return err
    
    # Try to find a low-rank decomposition using random initialization and local search
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 36
    
    # Schoolbook as fallback
    U_school = np.zeros((n * m, best_rank))
    V_school = np.zeros((m * p, best_rank))
    W_school = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try ranks from 30 down to 20
    for target_rank in range(30, 19, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Simple local search: try to reduce error
            err = error(U, V, W)
            for step in range(100):
                if time.time() > deadline - 30:
                    break
                
                # Pick a random element to perturb
                mat = random.choice([U, V, W])
                row = random.randint(0, mat.shape[0] - 1)
                col = random.randint(0, mat.shape[1] - 1)
                
                old_val = mat[row, col]
                mat[row, col] = old_val + random.gauss(0, 0.1)
                new_err = error(U, V, W)
                
                if new_err < err:
                    err = new_err
                else:
                    mat[row, col] = old_val
            
            if err < 1e-6:
                # Try to round to rationals
                U_rounded = np.round(U * 4) / 4
                V_rounded = np.round(V * 4) / 4
                W_rounded = np.round(W * 4) / 4
                
                if verify(U_rounded, V_rounded, W_rounded, tol=1e-6):
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    if best_rank < n * m * p:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
