import time
import numpy as np

DEADLINE = 150.0
START = time.time()

def _remaining():
    return DEADLINE - (time.time() - START)

def _build_seed_36():
    n, m, p = 2, 4, 5
    strassen_U = [
        [1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],
        [1,1,0,0],[-1,0,1,0],[0,1,0,-1],
    ]
    strassen_V = [
        [1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],
        [0,0,0,1],[1,1,0,0],[0,0,1,1],
    ]
    strassen_W = [
        [1,0,0,1],[0,1,0,-1],[0,0,1,1],[1,1,0,0],
        [-1,0,1,0],[0,0,0,1],[1,0,0,0],
    ]
    rank = 4*7 + 8
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for j0 in (0,2):
        for k0 in (0,2):
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        U[li*m + (j0+lj)][col] = strassen_U[r][li*2+lj]
                for lj in range(2):
                    for lk in range(2):
                        V[(j0+lj)*p + (k0+lk)][col] = strassen_V[r][lj*2+lk]
                for lk in range(2):
                    for li in range(2):
                        W[(k0+lk)*n + li][col] = strassen_W[r][lk*2+li]
                col += 1
    for i in range(n):
        for j in range(m):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
    return U, V, W

def _build_rank_32():
    n, m, p = 2, 4, 5
    strassen_U = [
        [1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],
        [1,1,0,0],[-1,0,1,0],[0,1,0,-1],
    ]
    strassen_V = [
        [1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],
        [0,0,0,1],[1,1,0,0],[0,0,1,1],
    ]
    strassen_W = [
        [1,0,0,1],[0,1,0,-1],[0,0,1,1],[1,1,0,0],
        [-1,0,1,0],[0,0,0,1],[1,0,0,0],
    ]
    # 4 blocks of 2x2x2 Strassen for the 2x4x4 part
    # j-blocks: 0, 2
    # k-blocks: 0, 2
    rank = 4*7 + 4
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for j0 in (0, 2):
        for k0 in (0, 2):
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        U[li*m + (j0+lj)][col] = strassen_U[r][li*2+lj]
                for lj in range(2):
                    for lk in range(2):
                        V[(j0+lj)*p + (k0+lk)][col] = strassen_V[r][lj*2+lk]
                for lk in range(2):
                    for li in range(2):
                        W[(k0+lk)*n + li][col] = strassen_W[r][lk*2+li]
                col += 1
    # 4 terms for the 5th column (k=4)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    for j in range(4):
        for i in range(2):
            # Each term is A[i,j] * B[j,4]
            # We need 4 terms, one for each j?
            # Actually, we can group by j.
            # For a fixed j, we have 2 terms (i=0, i=1).
            # But rank is additive.
            # Term for j=0: A[0,0]B[0,4] + A[1,0]B[1,4] (2 rank 1 terms)
            # Total 4*2 = 8 terms?
            # Wait, the 5th column part is 2x4x1.
            # Rank of 2x4x1 is 4?
            # C[i,4] = sum_j A[i,j] B[j,4].
            # This is a sum of 4 rank-1 tensors (for each j).
            # So 4 terms.
            pass
    
    # Correct construction for 5th column:
    # We need to add 4 terms.
    # Term r (0..3) corresponds to j=r.
    # U[i, r, col] = delta_{i, i} ? No.
    # T[i, j, 4] = 1 if j==r?
    # C[i, 4] = sum_j A[i,j] B[j,4].
    # Decomposition: sum_{r=0}^3 U[i, r, 7+r] V[r, 4, 7+r] W[4, i, 7+r].
    # We want sum_r U[i, r, 7+r] V[r, 4, 7+r] W[4, i, 7+r] = A[i, r] B[r, 4].
    # So U[i, r, 7+r] = 1, V[r, 4, 7+r] = 1, W[4, i, 7+r] = 1.
    for r in range(4):
        for i in range(2):
            U[i*m + r][col] = 1
            V[r*p + 4][col] = 1
            W[4*n + i][col] = 1
        col += 1
        
    return U, V, W

def _verify(U, V, W):
    n, m, p = 2, 4, 5
    rank = len(U[0])
    U = np.array(U, dtype=float); V = np.array(V, dtype=float); W = np.array(W, dtype=float)
    T = np.zeros((n*m, m*p, p*n))
    Texp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[(i*m+j),(j*p+k),(k*n+i)] += u[i,j]*v[j,k]*w[k,i]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Texp[(i*m+j),(j*p+k),(k*n+i)] = 1.0
    return np.allclose(T, Texp, atol=1e-9)

def construct():
    n, m, p = 2, 4, 5
    U32, V32, W32 = _build_rank_32()
    if _verify(U32, V32, W32):
        return {"U": U32, "V": V32, "W": W32}
    
    U, V, W = _build_seed_36()
    return {"U": U, "V": V, "W": W}
