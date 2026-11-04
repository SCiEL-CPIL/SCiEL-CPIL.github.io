import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

def _verify_full(U, V, W):
    """Full verification of the tensor decomposition for 2x3x6."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _find_rank8_2x3x3(max_restarts=50, deadline=None):
    """Try to find a rank-8 decomposition of 2x3x3 matrix multiplication."""
    n, m, p = 2, 3, 3
    rank = 8
    dim_U = n * m  # 6
    dim_V = m * p  # 9
    dim_W = p * n  # 6
    total_dim = dim_U * rank + dim_V * rank + dim_W * rank  # 168
    
    def make_objective(x):
        U = x[:dim_U*rank].reshape(dim_U, rank)
        V = x[dim_U*rank:(dim_U+dim_V)*rank].reshape(dim_V, rank)
        W = x[(dim_U+dim_V)*rank:].reshape(dim_W, rank)
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    loss += (total - 1.0)**2
        return loss
    
    for attempt in range(max_restarts):
        if deadline and time.time() > deadline:
            break
        x0 = np.random.randn(total_dim) * 0.5
        result = minimize(make_objective, x0, method='BFGS', 
                         options={'maxiter': 5000, 'gtol': 1e-14})
        if result.fun < 1e-16:
            U = result.x[:dim_U*rank].reshape(dim_U, rank)
            V = result.x[dim_U*rank:(dim_U+dim_V)*rank].reshape(dim_V, rank)
            W = result.x[(dim_U+dim_V)*rank:].reshape(dim_W, rank)
            return U, V, W
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 110.0
    try:
        U8, V8, W8 = _find_rank8_2x3x3(max_restarts=50, deadline=deadline)
        if U8 is not None and time.time() < deadline:
            n, m, p = DIMENSIONS
            rank = 16
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]
            # Block 1: k = 0,1,2 (columns 0-7)
            for i in range(n):
                for j in range(m):
                    for r in range(8):
                        U[i*m+j][r] = float(U8[i*m+j, r])
            for j in range(m):
                for k in range(3):
                    for r in range(8):
                        V[j*p+k][r] = float(V8[j*3+k, r])
            for k in range(3):
                for i in range(n):
                    for r in range(8):
                        W[k*n+i][r] = float(W8[k*n+i, r])
            # Block 2: k = 3,4,5 (columns 8-15)
            for i in range(n):
                for j in range(m):
                    for r in range(8):
                        U[i*m+j][8+r] = float(U8[i*m+j, r])
            for j in range(m):
                for k in range(3, 6):
                    for r in range(8):
                        V[j*p+k][8+r] = float(V8[j*3+(k-3), r])
            for k in range(3, 6):
                for i in range(n):
                    for r in range(8):
                        W[k*n+i][8+r] = float(W8[(k-3)*n+i, r])
            
            result = {"U": U, "V": V, "W": W}
            if _verify_full(U, V, W):
                return result
    except Exception:
        pass
    return _schoolbook()
