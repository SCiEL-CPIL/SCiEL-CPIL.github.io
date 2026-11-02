import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7

    start_time = time.time()
    deadline = start_time + 115.0

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > tol:
                            return False
        return True

    # Strategy: Rank 50 using Strassen blocks
    # A is 2x4. B is 4x7.
    # Split A cols: 2, 2
    # Split B cols: 2, 2, 2, 1
    # For each A block (2x2) and B block (2x2): Strassen (rank 7)
    # For each A block (2x2) and B block (2x1): Schoolbook (rank 4)
    
    rank = 50
    U = np.zeros((n*m, rank), dtype=np.int32)
    V = np.zeros((m*p, rank), dtype=np.int32)
    W = np.zeros((p*n, rank), dtype=np.int32)
    col = 0

    def add_strassen_222(i0, j0, k0):
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)

        U[a, col] += 1; U[d, col] += 1
        V[e, col] += 1; V[h, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        
        U[c, col] += 1; U[d, col] += 1
        V[e, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        
        U[a, col] += 1
        V[f, col] += 1; V[h, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        
        U[d, col] += 1
        V[g, col] += 1; V[e, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        
        U[a, col] += 1; U[b, col] += 1
        V[h, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        
        U[c, col] += 1; U[a, col] -= 1
        V[e, col] += 1; V[f, col] += 1
        W[c11, col] += 1
        col += 1
        
        U[b, col] += 1; U[d, col] -= 1
        V[g, col] += 1; V[h, col] += 1
        W[c00, col] += 1
        col += 1

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)

        U[a, col] += 1; V[e, col] += 1; W[c00, col] += 1; col += 1
        U[b, col] += 1; V[g, col] += 1; W[c00, col] += 1; col += 1
        U[c, col] += 1; V[e, col] += 1; W[c10, col] += 1; col += 1
        U[d, col] += 1; V[g, col] += 1; W[c10, col] += 1; col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_222(0, j_start, k_start)
        add_schoolbook_221(0, j_start, 6)

    if col != rank:
        # Fallback
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                    idx += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    if verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # Fallback
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1.0
                V[j*p + k, idx] = 1.0
                W[k*n + i, idx] = 1.0
                idx += 1
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
