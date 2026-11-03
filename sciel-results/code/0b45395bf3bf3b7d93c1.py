"""Search for low-rank tensor decomposition of 3x3 * 3x8 matrix multiplication."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 8]
START_TIME = time.time()
TIME_LIMIT = 145.0

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def _build_tensor(n, m, p):
    T = np.ones((n, m, p))
    return T

def _decompose_fast(U, V, W, n, m, p):
    """Fast decomposition using einsum-like approach."""
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    r = U.shape[1]
    result = np.zeros((n, m, p))
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                result[i, j, k] = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
    return result

def _verify(U, V, W, T, n, m, p, tol=1e-8):
    result = _decompose_fast(U, V, W, n, m, p)
    return np.max(np.abs(result - T)) < tol

def _to_fraction_list(arr):
    result = []
    for row in arr:
        r = []
        for val in row:
            f = Fraction(val).limit_denominator(1000000)
            r.append(f)
        result.append(r)
    return result

def _schoolbook(n, m, p):
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

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    
    # Search for lower rank
    for target_rank in range(24, best_rank + 1):
        if _time_left() < 8:
            break
        
        found = False
        max_iter = max(200, int(_time_left() / 0.02))
        
        for attempt in range(max_iter):
            if _time_left() < 5:
                break
            
            # Random sparse initialization
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            for r in range(target_rank):
                for _ in range(random.randint(1, 4)):
                    idx = random.randint(0, n * m - 1)
                    U[idx, r] = random.choice([-1, 1])
                for _ in range(random.randint(1, 4)):
                    idx = random.randint(0, m * p - 1)
                    V[idx, r] = random.choice([-1, 1])
                for _ in range(random.randint(1, 4)):
                    idx = random.randint(0, p * n - 1)
                    W[idx, r] = random.choice([-1, 1])
            
            if _verify(U, V, W, T, n, m, p):
                best_rank = target_rank
                best_result = {
                    "U": _to_fraction_list(U),
                    "V": _to_fraction_list(V),
                    "W": _to_fraction_list(W)
                }
                found = True
                break
        
        if found:
            break
    
    return best_result
