import time
import numpy as np

DEADLINE = 150.0
START = time.time()

def _remaining():
    return DEADLINE - (time.time() - START)

def _build_rank_40():
    n, m, p = 2, 4, 5
    rank = 40
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
    return U, V, W

def _build_rank_36():
    n, m, p = 2, 4, 5
    # Strassen decomposition for 2x2x2 blocks
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
    
    # 4 blocks of 2x2x2 (rank 7 each) + 8 terms for the 5th column
    rank = 4*7 + 8
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # Handle j in {0,1} and {2,3}
    for j0 in (0, 2):
        # Handle k in {0,1} and {2,3}
        for k0 in (0, 2):
            for r in range(7):
                # U part: 2x2 block
                for li in range(2):
                    for lj in range(2):
                        U[li*m + (j0+lj)][col] = strassen_U[r][li*2+lj]
                # V part: 2x2 block
                for lj in range(2):
                    for lk in range(2):
                        V[(j0+lj)*p + (k0+lk)][col] = strassen_V[r][lj*2+lk]
                # W part: 2x2 block
                for lk in range(2):
                    for li in range(2):
                        W[(k0+lk)*n + li][col] = strassen_W[r][lk*2+li]
                col += 1
        
    # Handle k=4 (last column of B)
    # This requires 8 terms: A[i,j] * B[j,4]
    for i in range(n):
        for j in range(m):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
            
    return U, V, W

def _verify(U, V, W):
    n, m, p = 2, 4, 5
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w).reshape(n*m, m*p, p*n)
        
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Texp[i*m+j, j*p+k, k*n+i] = 1.0
                
    return np.allclose(T, Texp, atol=1e-9)

def construct():
    n, m, p = 2, 4, 5
    
    # Try Rank 36 first
    try:
        U, V, W = _build_rank_36()
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
        
    # Fallback to Rank 40
    U, V, W = _build_rank_40()
    return {"U": U, "V": V, "W": W}
