import numpy as np
import time

def construct():
    n, m, p = 2, 2, 16
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Strassen coefficients for 2x2x2
    ms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 1, 0, 0]),
        ([1, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 0, 1], [1, 0, 1, 0], [0, 0, 0, 0]),
        ([0, 1, 0, -1], [0, 1, 0, 1], [-1, 0, 0, 1])
    ]
    
    # Verify Strassen block
    T = np.zeros((2, 2, 2))
    for r in range(7):
        u, v, w = ms[r]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    T[i,j,k] += u[i*2+j] * v[j*2+k] * w[k*2+i]
    
    if not np.allclose(T, 1):
        # Fallback to schoolbook
        rank = 64
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Construct for 2x2x16 using 4 blocks of 2x2x2 (rank 28)
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for s in range(4):
        for r in range(7):
            u, v, w = ms[r]
            idx = s*7 + r
            for i in range(2):
                for j in range(2):
                    U[i*m+j, idx] = u[i*2+j]
            for j in range(2):
                for k in range(2):
                    V[j*p + (2*s+k), idx] = v[j*2+k]
            for k in range(2):
                for i in range(2):
                    W[(2*s+k)*n + i, idx] = w[k*2+i]
    
    # Verify the full decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1 for all i,j,k
    # Actually for matrix multiplication, T[i,j,k] = 1 for all valid i,j,k
    T_full = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_full[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    if not np.allclose(T_full, 1):
        # Fallback to schoolbook
        rank = 64
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
