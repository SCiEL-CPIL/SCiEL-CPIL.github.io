import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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

def _search_rank_23(deadline):
    n, m, p = DIMENSIONS
    rank = 23
    
    best_U, best_V, best_W = None, None, None
    best_error = float('inf')
    
    max_restarts = 10
    restart = 0
    
    while time.time() < deadline - 5 and restart < max_restarts:
        restart += 1
        
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        lr = 0.01
        for step in range(5000):
            if time.time() > deadline - 5:
                break
            
            error = 0.0
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = U[i*m+j]
                        v = V[j*p+k]
                        w = W[k*n+i]
                        s = np.dot(u, v * w)
                        err = s - 1.0
                        error += err * err
                        dU[i*m+j] += 2.0 * err * (v * w)
                        dV[j*p+k] += 2.0 * err * (u * w)
                        dW[k*n+i] += 2.0 * err * (u * v)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
                if error < 1e-12:
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                        return U_r.tolist(), V_r.tolist(), W_r.tolist()
            
            if step % 1000 == 999:
                lr *= 0.8
    
    if best_U is not None and best_error < 1e-6:
        U_r = np.round(best_U * 2) / 2
        V_r = np.round(best_V * 2) / 2
        W_r = np.round(best_W * 2) / 2
        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def construct():
    overall_deadline = time.time() + 140
    result = _search_rank_23(overall_deadline)
    if result is not None:
        U, V, W = result
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    return _construct_schoolbook()
