"""Find rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Try to find a lower-rank decomposition
    for rank in range(30, 37):
        result = _try_rank(n, m, p, rank)
        if result is not None:
            return result
    
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for l in range(n):
            expected = 1.0 if i == l else 0.0
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+l, :])
                    if abs(val - expected) > 1e-8:
                        return False
    return True

def _try_rank(n, m, p, rank):
    """Try to find a valid decomposition of given rank."""
    np.random.seed(123)
    
    for attempt in range(100):
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Simple coordinate descent
        for it in range(10000):
            max_res = 0
            max_tuple = None
            for i in range(n):
                for l in range(n):
                    expected = 1.0 if i == l else 0.0
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+l, :])
                            res = val - expected
                            if abs(res) > abs(max_res):
                                max_res = res
                                max_tuple = (i, l, j, k)
            
            if abs(max_res) < 1e-10:
                if _validate(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
            
            if max_tuple is None:
                break
            
            i, l, j, k = max_tuple
            expected = 1.0 if i == l else 0.0
            res = max_res
            
            c = np.random.randint(0, rank)
            u = U[i*m+j, c]
            v = V[j*p+k, c]
            w = W[k*n+l, c]
            
            if abs(v * w) > 1e-10:
                U[i*m+j, c] += -res / (v * w)
            elif abs(u * w) > 1e-10:
                V[j*p+k, c] += -res / (u * w)
            elif abs(u * v) > 1e-10:
                W[k*n+l, c] += -res / (u * v)
            else:
                U[i*m+j, c] += np.random.randn() * 0.01
        
        if _validate(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None
