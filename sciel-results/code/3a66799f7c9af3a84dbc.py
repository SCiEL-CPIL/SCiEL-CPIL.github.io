import time
import numpy as np
import random

DIMENSIONS = [3, 4, 5]

def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Schoolbook
    rank0 = n * m * p
    U = np.zeros((n * m, rank0), dtype=float)
    V = np.zeros((m * p, rank0), dtype=float)
    W = np.zeros((p * n, rank0), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank0
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    np.random.seed(42)
    
    # Try to find lower rank by solving the system
    # For rank r, set up equations and solve
    for r in range(rank0 - 1, max(rank0 - 10, 1), -1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for attempt in range(10):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            U_r = np.random.randint(0, 2, size=(n * m, r)).astype(float)
            V_r = np.random.randint(0, 2, size=(m * p, r)).astype(float)
            W_r = np.random.randint(0, 2, size=(p * n, r)).astype(float)
            
            if verify_decomposition(U_r, V_r, W_r, n, m, p):
                best_rank = r
                best_U, best_V, best_W = U_r, V_r, W_r
                break
        
        if best_rank == r:
            break
    
    def to_list(arr):
        return [[float(round(x * 2) / 2) for x in row] for row in arr]
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
