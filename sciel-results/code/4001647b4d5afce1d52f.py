import time, random
import numpy as np

DIMENSIONS = [4, 4, 7]
_START = time.time()
_DEADLINE = 120.0
_MARGIN = 5.0


def _build_4x4x4():
    def lb(I, J):
        v = np.zeros(16)
        for di in range(2):
            for dj in range(2):
                v[(2 * I + di) * 4 + (2 * J + dj)] = 1
        return v
    def rb(K, L):
        v = np.zeros(16)
        for di in range(2):
            for dj in range(2):
                v[(2 * K + di) * 4 + (2 * L + dj)] = 1
        return v
    A = np.array([[lb(0,0), lb(0,1)], [lb(1,0), lb(1,1)]])
    B = np.array([[lb(0,2), lb(0,3)], [lb(1,2), lb(1,3)]])
    C = np.array([[lb(2,0), lb(2,1)], [lb(3,0), lb(3,1)]])
    D = np.array([[lb(2,2), lb(2,3)], [lb(3,2), lb(3,3)]])
    E = np.array([[rb(0,0), rb(0,1)], [rb(1,0), rb(1,1)]])
    F = np.array([[rb(0,2), rb(0,3)], [rb(1,2), rb(1,3)]])
    G = np.array([[rb(2,0), rb(2,1)], [rb(3,0), rb(3,1)]])
    H = np.array([[rb(2,2), rb(2,3)], [rb(3,2), rb(3,3)]])
    prods = [(A - D, E + H), (A + B, E), (D, F - G), (C, G + H),
             (A + C, H), (D - C, E + G), (B - D, E + F)]
    outmap = {
        (0,0): [(0,1),(3,1),(4,1),(5,1)],
        (0,1): [(1,1),(2,1)],
        (1,0): [(3,1),(6,1)],
        (1,1): [(0,1),(1,-1),(2,1),(6,1)],
    }
    U = np.zeros((16,49)); V = np.zeros((16,49)); W = np.zeros((16,49))
    col = 0
    for pi,(L,R) in enumerate(prods):
        lefts = [L[0,0]-L[1,1], L[0,0]+L[0,1], L[1,1], L[1,0],
                 L[0,0]+L[1,0], L[1,1]-L[1,0], L[0,1]-L[1,1]]
        rights = [R[0,0]+R[1,1], R[0,0], R[0,1]-R[1,0], R[1,0]+R[1,1],
                  R[1,1], R[0,0]+R[1,0], R[0,0]+R[0,1]]
        for t in range(7):
            U[:,col] = lefts[t].ravel()
            V[:,col] = rights[t].ravel()
            for (a,b),terms in outmap.items():
                for (pt,cf) in terms:
                    if pt == t:
                        for di in range(2):
                            for dj in range(2):
                                W[(2*a+di)*4+(2*b+dj), col] += cf
            col += 1
    return U, V, W


def _build_4x4x3_local():
    U = np.zeros((16,48)); V = np.zeros((12,48)); W = np.zeros((12,48))
    c = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i*4+j, c] = 1
                V[j*3+k, c] = 1
                W[k*3+i, c] = 1
                c += 1
    return U, V, W


def _verify(U, V, W, trials=4, tol=1e-6):
    n, m, p = DIMENSIONS
    rng = random.Random(12345)
    for _ in range(trials):
        X = np.array([[rng.uniform(-1,1) for _ in range(m)] for _ in range(n)])
        Y = np.array([[rng.uniform(-1,1) for _ in range(p)] for _ in range(m)])
        Z = np.zeros((n,p))
        rank = U.shape[1]
        for c in range(rank):
            A = U[:,c].reshape(n,m)
            B = V[:,c].reshape(m,p)
            C = W[:,c].reshape(p,n)
            Z += C * (A @ B)
        if np.max(np.abs(Z - X @ Y)) > tol:
            return False
    return True


def _to_lists(U, V, W):
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
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
    n, m, p = DIMENSIONS
    if time.time() - _START > _DEADLINE - _MARGIN:
        return _schoolbook()
    U1, V1, W1 = _build_4x4x4()
    U2, V2, W2 = _build_4x4x3_local()
    rank = 97
    U = np.zeros((16, rank)); V = np.zeros((28, rank)); W = np.zeros((28, rank))
    U[:, 0:49] = U1
    for j in range(4):
        for k in range(4):
            V[j*7+k, 0:49] = V1[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*7+i, 0:49] = W1[k*4+i, 0:49]
    U[:, 49:97] = U2
    for j in range(4):
        for k in range(3):
            V[j*7+k, 49:97] = V2[j*3+k, 49:97]
    for k in range(3):
        for i in range(4):
            W[k*7+i, 49:97] = W2[k*3+i, 49:97]
    if _verify(U, V, W):
        return _to_lists(U, V, W)
    return _schoolbook()
