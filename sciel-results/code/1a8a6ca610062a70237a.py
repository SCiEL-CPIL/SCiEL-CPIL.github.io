import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 8]

def _verify(U, V, W, n=2, m=4, p=8, trials=200, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for c in range(rank):
            factor_A = np.zeros(n)
            for i in range(n):
                factor_A[i] = sum(A[i, j] * U[i * m + j][c] for j in range(m))
            
            factor_B = np.zeros(p)
            for k in range(p):
                factor_B[k] = sum(V[j * p + k][c] * B[j, k] for j in range(m))
            
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += factor_A[i] * factor_B[k] * W[k * n + i][c]
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def _construct_strassen_56():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_strassen = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]
    ]
    V_strassen = [
        [1, 0, 0, 1], [0, 0, 0, 1], [1, 0, -1, 0], [0, -1, 0, 1],
        [0, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0], [1, -1, 1, 0, 0, 1, 0]
    ]
    
    A1_idx = [0, 1, 4, 5]
    A2_idx = [2, 3, 6, 7]
    
    B_top = [[0*8+k, 0*8+k+1, 1*8+k, 1*8+k+1] for k in range(0, 8, 2)]
    B_bot = [[2*8+k, 2*8+k+1, 3*8+k, 3*8+k+1] for k in range(0, 8, 2)]
    
    C_idx = [[k*2+i for i in range(2)] for k in range(8)]
    # C_idx is list of 8 lists of 2 indices? No, W index is k*n+i.
    # We need 4 indices per block: c00, c01, c10, c11.
    # For block col k, k+1.
    # Indices: k*2+0, k*2+1, (k+1)*2+0, (k+1)*2+1.
    
    blocks = []
    for i in range(4):
        b_top = [0*8+2*i, 0*8+2*i+1, 1*8+2*i, 1*8+2*i+1]
        b_bot = [2*8+2*i, 2*8+2*i+1, 3*8+2*i, 3*8+2*i+1]
        c_idx = [2*i*2+0, 2*i*2+1, (2*i+1)*2+0, (2*i+1)*2+1]
        
        blocks.append((A1_idx, b_top, c_idx))
        blocks.append((A2_idx, b_bot, c_idx))
        
    col_offset = 0
    for a_idx, b_idx, c_idx in blocks:
        for f in range(7):
            col = col_offset + f
            for r in range(4):
                U[a_idx[r]][col] = U_strassen[f][r]
                V[b_idx[r]][col] = V_strassen[f][r]
                W[c_idx[r]][col] = W_strassen[r][f]
        col_offset += 7
        
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to find a better solution?
    # For now, return the known 56 solution.
    # If we had a rank 52 solution, we would return it.
    
    sol = _construct_strassen_56()
    
    if _verify(sol["U"], sol["V"], sol["W"], trials=10):
        return sol
    else:
        # Fallback
        n, m, p = 2, 4, 8
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
