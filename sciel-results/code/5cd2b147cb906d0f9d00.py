import time
import numpy as np

_DIMS = (3, 3, 8)
_DEADLINE = 140.0
_T0 = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def _verify(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U_arr[:, r].reshape(n, m)
        v = V_arr[:, r].reshape(m, p)
        w = W_arr[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return bool(np.all(np.abs(T - 1.0) < tol))

def _strassen_2x2_2():
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = _DIMS
    # 4 Strassen blocks for i in {0,1}, j in {0,1}
    # k pairs: (0,1), (2,3), (4,5), (6,7)
    blocks = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
    
    rank_strassen = 7 * len(blocks)
    rank_rem = 8
    total_rank = rank_strassen + rank_rem
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    for (i0, j0, k0) in blocks:
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj), col+c] = sU[di*2+dj, c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk), col+c] = sV[dj*2+dk, c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di), col+c] = sW[dk*2+di, c]
        col += 7
        
    # Remainder: rank 8
    # i in {0,1,2}, j in {0,1,2}, k in {0..7}
    # Covered by Strassen: i in {0,1}, j in {0,1}
    # Remainder: (i=2, any j) or (any i, j=2)
    
    # r = 0..7
    for r in range(8):
        k = r
        # U: (i, j) -> i*m + j
        # S1: i in {0,1}, j=2
        U[0*m + 2, col+r] = 1
        U[1*m + 2, col+r] = 1
        # S2: i=2, j in {0,1}
        U[2*m + 0, col+r] = 1
        U[2*m + 1, col+r] = 1
        # S3: i=2, j=2
        U[2*m + 2, col+r] = 1
        
        # V: (j, k) -> j*p + k
        # S1, S3: j=2, k=r
        V[2*p + k, col+r] = 1
        # S2: j in {0,1}, k=r
        V[0*p + k, col+r] = 1
        V[1*p + k, col+r] = 1
        
        # W: (k, i) -> k*n + i
        # S1: k=r, i in {0,1}
        W[k*n + 0, col+r] = 1
        W[k*n + 1, col+r] = 1
        # S2, S3: k=r, i=2
        W[k*n + 2, col+r] = 1
        
    if not _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        # Fallback
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
