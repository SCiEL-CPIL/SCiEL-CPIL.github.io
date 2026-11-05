import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

def _matmul_tensor(n, m, p):
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            T[i, j, i] = 1
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', uc, vc, wc)
    expected = _matmul_tensor(n, m, p).astype(np.float64)
    return np.allclose(T, expected, atol=tol, rtol=0)

def _strassen_222():
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    # c0 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # c1 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    # c2 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    # c3 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    # c4 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    # c5 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # c6 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = -1
    return U, V, W

def _place_block2(U_full, V_full, W_full, U_blk, V_blk, W_blk,
                  n_full, m_full, p_full,
                  n_b, m_b, p_b, i0, j0, k0, c0):
    for i in range(n_b):
        for j in range(m_b):
            for c in range(U_blk.shape[1]):
                U_full[(i0 + i) * m_full + (j0 + j), c0 + c] += U_blk[i * m_b + j, c]
    for j in range(m_b):
        for k in range(p_b):
            for c in range(V_blk.shape[1]):
                V_full[(j0 + j) * p_full + (k0 + k), c0 + c] += V_blk[j * p_b + k, c]
    for k in range(p_b):
        for i in range(n_b):
            for c in range(W_blk.shape[1]):
                W_full[(k0 + k) * n_full + (i0 + i), c0 + c] += W_blk[k * n_b + i, c]

def _schoolbook(n, m, p):
    r = n * m * p
    U = np.zeros((n * m, r), dtype=np.int64)
    V = np.zeros((m * p, r), dtype=np.int64)
    W = np.zeros((p * n, r), dtype=np.int64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _decompose_332():
    n, m, p = 3, 3, 2
    total_rank = 17
    U = np.zeros((n * m, total_rank), dtype=np.int64)
    V = np.zeros((m * p, total_rank), dtype=np.int64)
    W = np.zeros((p * n, total_rank), dtype=np.int64)
    
    U1, V1, W1 = _strassen_222()
    _place_block2(U, V, W, U1, V1, W1, n, m, p, 2, 2, 2, 0, 0, 0, 0)
    
    U2 = np.zeros((2*1, 4), dtype=np.int64)
    V2 = np.zeros((1*2, 4), dtype=np.int64)
    W2 = np.zeros((2*2, 4), dtype=np.int64)
    # Schoolbook 2x1x2
    c=0
    for i in range(2):
        for j in range(1):
            for k in range(2):
                U2[i*1+j, c] = 1
                V2[j*2+k, c] = 1
                W2[k*2+i, c] = 1
                c+=1
    _place_block2(U, V, W, U2, V2, W2, n, m, p, 2, 1, 2, 0, 2, 0, 7)
    
    U3 = np.zeros((1*2, 4), dtype=np.int64)
    V3 = np.zeros((2*2, 4), dtype=np.int64)
    W3 = np.zeros((2*1, 4), dtype=np.int64)
    # Schoolbook 1x2x2
    c=0
    for i in range(1):
        for j in range(2):
            for k in range(2):
                U3[i*2+j, c] = 1
                V3[j*2+k, c] = 1
                W3[k*1+i, c] = 1
                c+=1
    _place_block2(U, V, W, U3, V3, W3, n, m, p, 1, 2, 2, 2, 0, 0, 11)
    
    U4 = np.zeros((1*1, 2), dtype=np.int64)
    V4 = np.zeros((1*2, 2), dtype=np.int64)
    W4 = np.zeros((2*1, 2), dtype=np.int64)
    # Schoolbook 1x1x2
    c=0
    for i in range(1):
        for j in range(1):
            for k in range(2):
                U4[i*1+j, c] = 1
                V4[j*2+k, c] = 1
                W4[k*1+i, c] = 1
                c+=1
    _place_block2(U, V, W, U4, V4, W4, n, m, p, 1, 1, 2, 2, 2, 0, 15)
    
    return U, V, W

def _decompose_334():
    n, m, p = 3, 3, 4
    total_rank = 34
    U = np.zeros((n * m, total_rank), dtype=np.int64)
    V = np.zeros((m * p, total_rank), dtype=np.int64)
    W = np.zeros((p * n, total_rank), dtype=np.int64)
    
    U1, V1, W1 = _decompose_332()
    _place_block2(U, V, W, U1, V1, W1, n, m, p, 3, 3, 2, 0, 0, 0, 0)
    
    U2, V2, W2 = _decompose_332()
    _place_block2(U, V, W, U2, V2, W2, n, m, p, 3, 3, 2, 0, 0, 2, 17)
    
    return U, V, W

def _decompose_338_rank68():
    n, m, p = 3, 3, 8
    total_rank = 68
    U = np.zeros((n * m, total_rank), dtype=np.int64)
    V = np.zeros((m * p, total_rank), dtype=np.int64)
    W = np.zeros((p * n, total_rank), dtype=np.int64)
    
    U1, V1, W1 = _decompose_334()
    _place_block2(U, V, W, U1, V1, W1, n, m, p, 3, 3, 4, 0, 0, 0, 0)
    
    U2, V2, W2 = _decompose_334()
    _place_block2(U, V, W, U2, V2, W2, n, m, p, 3, 3, 4, 0, 0, 4, 34)
    
    return U, V, W

def construct():
    n, m, p = 3, 3, 8
    
    # Fallback: Rank 68
    U, V, W = _decompose_338_rank68()
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Schoolbook 72
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
