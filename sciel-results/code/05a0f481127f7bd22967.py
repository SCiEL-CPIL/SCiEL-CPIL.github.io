import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 120.0

    def time_left():
        return deadline - time.time()

    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def check_decomposition(U, V, W, T, tol=1e-6):
        T_rec = np.einsum('ir,vr,wr->ivw', U, V, W, optimize=True)
        return np.max(np.abs(T_rec - T)) < tol

    # Rank 28 decomposition (Strassen + Schoolbook)
    def build_rank_28():
        rank = 28
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0

        def add_strassen(i1, i2, j1, j2, k1, k2):
            nonlocal col
            U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1
            W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
            col += 1
            U[i1*m+j1, col] = 1
            V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
            W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            U[i2*m+j2, col] = 1
            V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
            W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
            col += 1
            U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
            V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
            col += 1
            U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
            V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
            W[k2*n+i2, col] = 1
            col += 1
            U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
            V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1
            col += 1

        def add_outer(i, j, k):
            nonlocal col
            U[i*m+j, col] = 1
            V[j*p+k, col] = 1
            W[k*n+i, col] = 1
            col += 1

        add_strassen(0, 1, 0, 1, 0, 1)
        add_strassen(0, 1, 0, 1, 2, 3)
        
        for i in range(n):
            for k in range(2):
                add_outer(i, 2, k)
        
        for j in range(2):
            for i in range(n):
                add_outer(i, j, 4)
        
        for i in range(n):
            for k in range(2, 5):
                add_outer(i, 2, k)

        assert col == rank
        return U, V, W

    U28, V28, W28 = build_rank_28()
    if check_decomposition(U28, V28, W28, T):
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
