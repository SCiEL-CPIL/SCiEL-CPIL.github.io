import numpy as np
from scipy.optimize import least_squares
import random
import time

DIMENSIONS = [2, 2, p]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    best = None
    best_rank = n * m * p
    
    # Try ranks from 22 to 27
    for rank in range(22, min(28, best_rank)):
        if time.time() - start_time > 100:
            break
        for attempt in range(20):
            if time.time() - start_time > 110:
                break
            try:
                result = attempt_decomposition(n, m, p, rank, attempt)
                if result is not None:
                    best = result
                    best_rank = rank
                    break
            except Exception:
                continue
        if best is not None:
            break
    
    if best is not None:
        return best
    else:
        return schoolbook(n, m, p)

def schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def attempt_decomposition(n, m, p, rank, seed):
    np.random.seed(seed)
    
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    def residuals(params):
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        res = np.zeros((n*m)*(m*p)*(p*n))
        idx = 0
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            for i2 in range(n):
                                val = 0
                                for t in range(rank):
                                    val += U[i*m+j, t] * V[j2*p+k, t] * W[k2*n+i2, t]
                                target = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                                res[idx] = val - target
                                idx += 1
        return res
    
    params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        result = least_squares(residuals, params0, method='lm', max_nfev=500)
        if np.max(np.abs(result.fun)) < 1e-6:
            U = result.x[:n*m*rank].reshape(n*m, rank)
            V = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except:
        pass
    
    return None
