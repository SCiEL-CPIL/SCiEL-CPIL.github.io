"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-8):
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                valid.add((a,b,c))
                s = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(s - 1.0) > tol:
                    return False
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a,b,c) not in valid:
                    s = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                    if abs(s) > tol:
                        return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = n * m * p
    
    # Quick search for lower rank
    import random
    random.seed(42)
    
    for target_rank in range(30, best_rank):
        if time.time() > deadline - 20:
            break
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            for it in range(30):
                if time.time() > deadline - 20:
                    break
                for c2 in range(p*n):
                    k = c2 // n
                    i = c2 % n
                    A_mat = np.zeros((m, target_rank))
                    b_vec = np.ones(m)
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    W[c2, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                for a in range(n*m):
                    i = a // m
                    j = a % m
                    A_mat = np.zeros((p, target_rank))
                    b_vec = np.ones(p)
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    U[a, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                for b in range(m*p):
                    j = b // p
                    k = b % p
                    A_mat = np.zeros((n, target_rank))
                    b_vec = np.ones(n)
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    V[b, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            U_l = U.tolist()
            V_l = V.tolist()
            W_l = W.tolist()
            if _verify(U_l, V_l, W_l, n, m, p, tol=1e-4):
                def rh(x):
                    return round(x*2)/2
                U_r = [[rh(v) for v in row] for row in U_l]
                V_r = [[rh(v) for v in row] for row in V_l]
                W_r = [[rh(v) for v in row] for row in W_l]
                if _verify(U_r, V_r, W_r, n, m, p, tol=1e-2):
                    return {"U": U_r, "V": V_r, "W": W_r}
                if target_rank < best_rank:
                    best = {"U": U_l, "V": V_l, "W": W_l}
                    best_rank = target_rank
                break
        if time.time() > deadline - 15:
            break
    
    return best
# EVOLVE-BLOCK-END
