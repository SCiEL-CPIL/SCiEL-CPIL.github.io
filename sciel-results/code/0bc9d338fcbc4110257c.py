import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _build_strassen_2x2():
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_4x4x4_strassen():
    U2, V2, W2 = _build_strassen_2x2()
    U = np.zeros((16, 49), dtype=np.int64)
    V = np.zeros((16, 49), dtype=np.int64)
    W = np.zeros((16, 49), dtype=np.int64)
    big_mults = [
        {'A': [((0,0),1),((1,1),1)], 'B': [((0,0),1),((1,1),1)], 'C': [((0,0),1),((1,1),1)]},
        {'A': [((1,0),1),((1,1),1)], 'B': [((0,0),1)], 'C': [((0,1),1),((1,1),-1)]},
        {'A': [((0,0),1)], 'B': [((0,1),1),((1,1),-1)], 'C': [((1,0),1),((1,1),1)]},
        {'A': [((1,1),1)], 'B': [((1,0),1),((0,0),-1)], 'C': [((0,0),1),((0,1),1)]},
        {'A': [((0,0),1),((0,1),1)], 'B': [((1,1),1)], 'C': [((0,0),-1),((1,0),1)]},
        {'A': [((1,0),1),((0,0),-1)], 'B': [((0,0),1),((0,1),1)], 'C': [((1,1),1)]},
        {'A': [((0,1),1),((1,1),-1)], 'B': [((1,0),1),((1,1),1)], 'C': [((0,0),1)]},
    ]
    for bi_, bm in enumerate(big_mults):
        for r2 in range(7):
            col = bi_ * 7 + r2
            for (bi,bj),sa in bm['A']:
                for i2 in range(2):
                    for k2 in range(2):
                        uv = U2[i2*2+k2, r2]
                        if uv:
                            U[(2*bi+i2)*4+(2*bj+k2), col] += sa*uv
            for (bk,bj),sb in bm['B']:
                for k2 in range(2):
                    for j2 in range(2):
                        vv = V2[k2*2+j2, r2]
                        if vv:
                            V[(2*bk+k2)*4+(2*bj+j2), col] += sb*vv
            for (bi,bj),sc in bm['C']:
                for i2 in range(2):
                    for j2 in range(2):
                        wv = W2[i2*2+j2, r2]
                        if wv:
                            W[(2*bi+i2)*4+(2*bj+j2), col] += sc*wv
    return U, V, W

def _build_4x4x2_rank16():
    """Rank-16 schoolbook for 4x4 * 4x2."""
    U = np.zeros((16, 16), dtype=np.int64)
    V = np.zeros((8, 16), dtype=np.int64)
    W = np.zeros((8, 16), dtype=np.int64)
    col = 0
    for i in range(4):
        for k in range(4):
            for j in range(2):
                U[i*4+k, col] = 1
                V[k*2+j, col] = 1
                W[j*4+i, col] = 1
                col += 1
    return U, V, W

def _schoolbook_fallback():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col]=1; V[j*p+k][col]=1; W[k*n+i][col]=1; col+=1
    return {"U": U, "V": V, "W": W}

def _validate_numpy(U, V, W, n, m, p):
    Un = np.asarray(U, dtype=np.int64)
    Vn = np.asarray(V, dtype=np.int64)
    Wn = np.asarray(W, dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if np.sum(Un[i*m+j] * Vn[j*p+k] * Wn[k*n+i]) != 1:
                    return False
    return True

def _build_4x4x9_strassen():
    """Rank-114: two 4x4x4 Strassen (49 each) + schoolbook 4x4x1 (16)."""
    U444, V444, W444 = _build_4x4x4_strassen()
    n, m, p = 4, 4, 9
    total_rank = 114
    U = np.zeros((n*m, total_rank), dtype=np.int64)
    V = np.zeros((m*p, total_rank), dtype=np.int64)
    W = np.zeros((p*n, total_rank), dtype=np.int64)
    U[:, 0:49] = U444
    for k in range(4):
        for j in range(4):
            V[k*p+j, 0:49] = V444[k*4+j, :]
    for k in range(4):
        for i in range(4):
            W[k*n+i, 0:49] = W444[k*4+i, :]
    off = 49
    U[:, off:off+49] = U444
    for k in range(4):
        for j in range(4):
            V[k*p+(j+4), off:off+49] = V444[k*4+j, :]
    for k in range(4):
        for i in range(4):
            W[(k+4)*n+i, off:off+49] = W444[k*4+i, :]
    off3 = 98
    for i in range(4):
        for k in range(4):
            U[i*4+k, off3+i*4+k] = 1
            V[k*4+8, off3+i*4+k] = 1
            W[8*4+i, off3+i*4+k] = 1
    return U, V, W

def construct():
    t0 = time.time()
    U, V, W = _build_4x4x9_strassen()
    if not _validate_numpy(U, V, W, 4, 4, 9):
        return _schoolbook_fallback()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
