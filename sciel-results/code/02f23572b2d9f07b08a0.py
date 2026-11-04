import time
import sys
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

def _build_T(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    T = _build_T(n, m, p)
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    recon = np.zeros_like(T)
    for r in range(rank):
        recon += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    return np.max(np.abs(recon - T)) < tol

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
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    T = _build_T(n, m, p)
    T_W = T.reshape(p*n, n*m*m*p)
    T_U = T.reshape(n*m, m*p*p*n)
    T_V = T.reshape(m*p, n*m*p*n)
    
    for target_rank in range(28, best_rank):
        if time.time() > deadline - 25:
            break
        found = False
        for attempt in range(15):
            if time.time() > deadline - 25:
                break
            np.random.seed(42 + attempt * 17)
            U = np.random.uniform(-1, 1, (n*m, target_rank))
            V = np.random.uniform(-1, 1, (m*p, target_rank))
            W = np.random.uniform(-1, 1, (p*n, target_rank))
            for it in range(500):
                if time.time() > deadline - 25:
                    break
                M = np.stack([(U[:,t] * V[:,t]).flatten() for t in range(target_rank)], axis=1)
                W_new, _, _, _ = np.linalg.lstsq(M, T_W.T, rcond=None)
                W = W_new.T
                N = np.stack([(V[:,t] * W[:,t]).flatten() for t in range(target_rank)], axis=1)
                U_new, _, _, _ = np.linalg.lstsq(N, T_U.T, rcond=None)
                U = U_new.T
                P = np.stack([(U[:,t] * W[:,t]).flatten() for t in range(target_rank)], axis=1)
                V_new, _, _, _ = np.linalg.lstsq(P, T_V.T, rcond=None)
                V = V_new.T
                if it % 20 == 0 or it == 499:
                    recon = np.zeros((n*m, m*p, p*n))
                    for t in range(target_rank):
                        recon += np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t])
                    if np.max(np.abs(recon - T)) < 1e-9:
                        found = True
                        break
            if found:
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                break
        if found:
            break
    return best
