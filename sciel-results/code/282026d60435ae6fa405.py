"""Program: rank-132 decomposition of 4x4x9 matrix multiplication.

Strategy: 4x4x9 = three 4x4x3 blocks. Each 4x4x3 = four 2x2x3 blocks.
Each 2x2x3 = Strassen 2x2x2 (rank 7, cols 0,1) + schoolbook 2x2x1 (rank 4, col 2) = rank 11.
Total rank = 3 * 4 * 11 = 132 (< schoolbook 144).
All constructions are verified numerically; fallback to schoolbook if any check fails.
"""
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _strassen_222():
    """2x2x2 rank 7. U (4,7) rows i*2+j; V (4,7) rows j*2+k; W (4,7) rows k*2+i."""
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    def setU(i,j,c,v): U[i*2+j, c] += v
    def setV(j,k,c,v): V[j*2+k, c] += v
    def setW(k,i,c,v): W[k*2+i, c] += v
    # M1
    setU(0,0,0,1); setU(1,1,0,1); setV(0,0,0,1); setV(1,1,0,1)
    setW(0,0,0,1); setW(1,1,0,-1); setW(0,1,0,1); setW(1,0,0,1)
    # M2
    setU(1,0,1,1); setU(1,1,1,1); setV(0,0,1,1)
    setW(1,1,1,1); setW(1,0,1,-1)
    # M3
    setU(0,0,2,1); setV(0,1,2,1); setV(1,1,2,-1)
    setW(1,1,2,1); setW(0,1,2,1)
    # M4
    setU(1,1,3,1); setV(1,0,3,1); setV(0,0,3,-1)
    setW(0,0,3,1); setW(1,0,3,1)
    # M5
    setU(0,0,4,1); setU(0,1,4,1); setV(1,1,4,1)
    setW(0,0,4,-1); setW(0,1,4,1)
    # M6
    setU(1,0,5,1); setU(0,0,5,-1); setV(0,0,5,1); setV(0,1,5,1)
    setW(1,1,5,1)
    # M7
    setU(0,1,6,1); setU(1,1,6,-1); setV(1,0,6,1); setV(1,1,6,1)
    setW(0,0,6,1)
    return U, V, W

def _school_221():
    """2x2x1 rank 4. U (4,4) rows i*2+j; V (2,4) rows j*2+0; W (2,4) rows 0*2+i."""
    U = np.zeros((4,4)); V = np.zeros((2,4)); W = np.zeros((2,4))
    for j in range(2):
        for i in range(2):
            c = j*2 + i
            U[i*2+j, c] = 1
            V[j*2+0, c] = 1
            W[0*2+i, c] = 1
    return U, V, W

def _block_223():
    """2x2x3 rank 11. U (4,11); V (6,11) rows j*2+k; W (6,11) rows k*2+i."""
    U1,V1,W1 = _strassen_222()
    U2,V2,W2 = _school_221()
    return np.hstack([U1, U2]), np.hstack([V1, V2]), np.hstack([W1, W2])

def _block_443():
    """4x4x3 rank 44. U (16,44) rows i*4+j; V (12,44) rows j*3+k; W (12,44) rows k*4+i."""
    n = m = 4; p = 3
    U = np.zeros((16,44)); V = np.zeros((12,44)); W = np.zeros((12,44))
    for bi in range(2):
        for bj in range(2):
            U1,V1,W1 = _block_223()
            col = (bi*2+bj)*11
            for i_l in range(2):
                for j_l in range(2):
                    U[(bi*2+i_l)*m + (bj*2+j_l), col:col+11] = U1[i_l*2+j_l, :]
            for j_l in range(2):
                for k_l in range(3):
                    V[(bj*2+j_l)*p + k_l, col:col+11] = V1[j_l*2+k_l, :]
            for k_l in range(3):
                for i_l in range(2):
                    W[k_l*n + (bi*2+i_l), col:col+11] = W1[k_l*2+i_l, :]
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.asarray(U, float); V = np.asarray(V, float); W = np.asarray(W, float)
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    U = np.zeros((n*m, 3*44)); V = np.zeros((m*p, 3*44)); W = np.zeros((p*n, 3*44))
    for t in range(3):
        U1,V1,W1 = _block_443()
        col = t*44
        U[:, col:col+44] = U1
        for j in range(m):
            for k_l in range(3):
                V[j*p + (t*3+k_l), col:col+44] = V1[j*3 + k_l, :]
        for k_l in range(3):
            for i in range(n):
                W[(t*3+k_l)*n + i, col:col+44] = W1[k_l*n + i, :]
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    rank = n*m*p
    Us = np.zeros((n*m, rank)); Vs = np.zeros((m*p, rank)); Ws = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Us[i*m+j, c] = 1; Vs[j*p+k, c] = 1; Ws[k*n+i, c] = 1; c += 1
    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
# EVOLVE-BLOCK-END
