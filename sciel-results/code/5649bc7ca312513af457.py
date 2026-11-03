import numpy as np
import time
import itertools

_START_TIME = time.time()
_TIME_LIMIT = 115.0

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    # Check invalid entries
    for x in range(p*n):
        i = x % n
        k = x // n
        for y in range(n*m):
            j = y % m
            i2 = y // m
            for z in range(m*p):
                j2 = z // p
                k2 = z % p
                if i == i2 and j == j2 and k == k2:
                    continue
                s = np.dot(U[y, :], V[z, :] * W[x, :])
                if abs(s) > tol:
                    return False
    return True

def _build_54_rank(n, m, p):
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # 7 Strassen terms
            # 1
            U[0*m + j0, col] += 1; U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1; V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1; W[k1*n + 1, col] += 1
            col += 1
            # 2
            U[1*m + j0, col] += 1; U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1; W[k1*n + 1, col] -= 1
            col += 1
            # 3
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1; V[j1*p + k1, col] -= 1
            W[k1*n + 0, col] += 1; W[k1*n + 1, col] += 1
            col += 1
            # 4
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1; V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1; W[k0*n + 1, col] += 1
            col += 1
            # 5
            U[0*m + j0, col] += 1; U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] -= 1; W[k1*n + 0, col] += 1
            col += 1
            # 6
            U[1*m + j0, col] += 1; U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1; V[j0*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            # 7
            U[0*m + j1, col] += 1; U[1*m + j1, col] -= 1
            V[j1*p + k0, col] += 1; V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            col += 1
    # j=4
    for k in range(p):
        U[0*m + 4, col] += 1; V[4*p + k, col] += 1; W[k*n + 0, col] += 1; col += 1
        U[1*m + 4, col] += 1; V[4*p + k, col] += 1; W[k*n + 1, col] += 1; col += 1
    return U, V, W

def construct():
    n, m, p = 2, 5, 6
    U, V, W = _build_54_rank(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = 54
    
    if not _verify(U, V, W, n, m, p):
        # Fallback
        rank = n*m*p
        U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1; V[j*p+k, col] = 1; W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Greedy elimination
    current_U, current_V, current_W = U, V, W
    current_rank = 54
    
    while _time_left() > 10:
        eliminated = False
        for r in range(current_rank):
            if _time_left() < 5: break
            mask = np.ones(current_rank, dtype=bool)
            mask[r] = False
            U_t = current_U[:, mask]; V_t = current_V[:, mask]; W_t = current_W[:, mask]
            if _verify(U_t, V_t, W_t, n, m, p):
                current_U, current_V, current_W = U_t, V_t, W_t
                current_rank -= 1
                eliminated = True
                break
        if not eliminated:
            break
            
    if current_rank < best_rank:
        best_U, best_V, best_W = current_U, current_V, current_W
        best_rank = current_rank

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
